#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""抓取**真实场景测试图**，用于端到端验证项目功能。

为什么要新工具（`tools/fetch_dataset.py` 不够用）
-------------------------------------------------
`fetch_dataset.py` 抓的是**单物体裁剪图**——YOLO 检出目标后把 ROI 切出来存盘。
那种图**完全绕过了检测环节**，只能测 ResNet18 分类，测不到项目真正的链路：

    相机帧 → YOLO 检测 → 选框 → 裁剪 ROI → ResNet18 → 特征库检索 → 未知判断 → 位置播报

本工具抓的是**整幅场景照**：物体只是画面的一部分，可能偏心、遮挡、多物体同框。
这才对应现场演示的真实输入。抓完仍用 YOLO 校验「目标物确实在画面里」，
否则测的是噪声。许可与出处照样逐条记录（沿用 fetch_dataset 的纪律）。

产出
----
    data/test_scenes/<key>_scene_<NN>.jpg      完整场景图（不裁剪）
    data/test_scenes/manifest.json             出处 / 许可 / YOLO 检出框

用法
----
    python tools/fetch_test_scenes.py --list
    python tools/fetch_test_scenes.py --per-class 3
    python tools/fetch_test_scenes.py --only cup,keyboard
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request
from typing import List, Optional

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT_ROOT = os.path.join("data", "test_scenes")
API = "https://commons.wikimedia.org/w/api.php"
UA = "cv-coursework-scene-test/1.0 (student course project)"

# 场景式 query —— 刻意找「物体在环境里」而不是「物体特写」。
# 与 fetch_dataset.py 的 query 风格相反：那里要干净特写，这里要含环境。
SCENES = [
    dict(key="cup",       cn="马克杯",     cls="cup",
         queries=["mug on wooden desk", "coffee cup on table with laptop",
                  "cup on office desk", "tea cup on table"]),
    dict(key="bottle",    cn="保温瓶",     cls="bottle",
         queries=["water bottle on desk", "thermos on table",
                  "metal bottle on wooden table", "bottle beside backpack"]),
    dict(key="keyboard",  cn="键盘",       cls="keyboard",
         queries=["keyboard on desk with monitor", "computer desk keyboard mouse",
                  "desktop computer keyboard", "keyboard in front of monitor"]),
    dict(key="scissors",  cn="剪刀",       cls="scissors",
         queries=["scissors on table", "scissors on craft desk",
                  "scissors with paper on desk", "scissors beside notebook"]),
    dict(key="clock",     cn="闹钟",       cls="clock",
         queries=["alarm clock on nightstand", "clock on desk",
                  "clock on table near window", "table clock on shelf"]),
    dict(key="laptop",    cn="笔记本电脑", cls="laptop",
         queries=["laptop on desk with coffee", "open laptop on table",
                  "laptop on wooden desk", "laptop workstation desk"]),
    # ---- unknown（未录入），用来验证「未知物体」判断 ----
    dict(key="unknown_bowl",     cn="碗",   cls="bowl",
         queries=["bowl on kitchen table", "ceramic bowl on wooden table",
                  "soup bowl on table"]),
    dict(key="unknown_banana",   cn="香蕉", cls="banana",
         queries=["bananas on table", "banana on kitchen counter",
                  "bananas on wooden table"]),
    dict(key="unknown_umbrella", cn="雨伞", cls="umbrella",
         queries=["umbrella leaning on wall", "umbrella on floor",
                  "closed umbrella on table"]),
    dict(key="unknown_chair",    cn="椅子", cls="chair",
         queries=["chair in room", "wooden chair on floor",
                  "office chair at desk"]),
    # ---- 反例：目标物不在画面里，验证检测不误报 ----
    dict(key="empty_scene", cn="无目标场景", cls=None,
         queries=["empty wooden table", "plain desk background",
                  "room without furniture", "empty shelf"]),
]


def _ssl_ctx() -> ssl.SSLContext:
    return ssl.create_default_context()


def api(params: dict, retries: int = 3) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, context=_ssl_ctx(), timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.0)
    raise RuntimeError(f"Commons API failed: {last}")


def search_files(query: str, limit: int = 30) -> List[dict]:
    """返回 fileinfo（含 url / width / height / extmetadata）。"""
    d = api({
        "action": "query",
        "generator": "search",
        "gsrsearch": f'filetype:bitmap {query}',
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "format": "json",
    })
    out = []
    for p in (d.get("query", {}).get("pages", {}) or {}).values():
        ii = (p.get("imageinfo") or [{}])[0]
        if not ii.get("url"):
            continue
        if ii.get("mime") not in ("image/jpeg", "image/png"):
            continue
        w, h = ii.get("width", 0), ii.get("height", 0)
        if w < 640 or h < 480:
            continue
        em = ii.get("extmetadata", {}) or {}
        out.append({
            "title": p.get("title", ""),
            "url": ii["url"],
            "width": w,
            "height": h,
            "license": (em.get("LicenseShortName", {}) or {}).get("value", "unknown"),
            "author": (em.get("Artist", {}) or {}).get("value", ""),
            "page": f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(p.get('title',''))}",
        })
    return out


def download(url: str, retries: int = 3) -> Optional[bytes]:
    """带退避的下载。Commons 对连续批量请求会限流（实测 160 次后开始返回失败），
    所以失败要等一会儿再试，并逐步拉长间隔。"""
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, context=_ssl_ctx(), timeout=60) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            if i < retries - 1:
                time.sleep(2.0 * (i + 1))
    return None


def decode(data: bytes) -> Optional[np.ndarray]:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def _first_sentence(html: str) -> str:
    import re

    txt = re.sub(r"<[^>]+>", "", html or "").strip()
    return txt[:80]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="抓取真实场景测试图（完整场景，不裁剪）")
    p.add_argument("--list", action="store_true", help="打印场景计划后退出")
    p.add_argument("--per-class", type=int, default=3, help="每个类别抓几张")
    p.add_argument("--only", default=None, help="只抓指定 key，逗号分隔")
    p.add_argument("--min-conf", type=float, default=0.25,
                   help="YOLO 最低置信度（场景图目标可能较小/被遮挡，阈值放宽）")
    p.add_argument("--min-box", type=int, default=60,
                   help="目标框最短边像素数，低于此视为过小")
    p.add_argument("--candidates", type=int, default=12, help="每个 query 取几张候选")
    p.add_argument("--out", default=OUT_ROOT)
    args = p.parse_args(argv)

    if args.list:
        for s in SCENES:
            print("%-16s %-12s cls=%-10s queries=%d"
                  % (s["key"], s["cn"], s["cls"], len(s["queries"])))
        return 0

    from vision.detector import YoloDetector, COCO_CLASSES
    from core.config import load_config

    ycfg = dict(load_config().get("yolo", {}) or {})
    ycfg["conf_threshold"] = float(args.min_conf)
    det = YoloDetector(ycfg)
    det.load()
    print(f"[ok] YOLO {det.device_used} conf_th={det.conf_th}")

    os.makedirs(args.out, exist_ok=True)
    only = {x.strip() for x in args.only.split(",")} if args.only else None

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "Wikimedia Commons",
        "purpose": "scene-level end-to-end test (full image, NOT cropped)",
        "yolo": det.device_used,
        "min_conf": args.min_conf,
        "scenes": [],
    }
    mp = os.path.join(args.out, "manifest.json")
    if os.path.isfile(mp):
        try:
            manifest = json.load(open(mp, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass

    for spec in SCENES:
        key = spec["key"]
        if only and key not in only:
            continue
        cls_name = spec["cls"]
        want_cls_id = COCO_CLASSES.index(cls_name) if cls_name else None
        got = 0
        n_try = 0
        entry = dict(key=key, cn=spec["cn"], cls=cls_name,
                     cls_id=want_cls_id, images=[])

        for q in spec["queries"]:
            if got >= args.per_class:
                break
            try:
                cands = search_files(q, limit=args.candidates)
            except Exception as e:  # noqa: BLE001
                print(f"[{key}] search failed '{q}': {e}")
                continue
            for c in cands:
                if got >= args.per_class:
                    break
                n_try += 1
                raw = download(c["url"])
                if not raw:
                    continue
                img = decode(raw)
                if img is None:
                    continue
                h, w = img.shape[:2]

                dets, _dt = det.infer(img)
                if cls_name is None:
                    # 反例：目标类不应出现
                    hit = [d for d in dets if d.cls_id == want_cls_id]
                    if hit:
                        continue  # 有人把目标物拍进去了，换一张
                    status = "no_target_ok"
                    box = None
                    conf = 0.0
                else:
                    cand = [d for d in dets if d.cls_id == want_cls_id
                            and d.conf >= args.min_conf]
                    if not cand:
                        continue
                    d = max(cand, key=lambda x: x.conf)
                    x1, y1, x2, y2 = map(int, d.xyxy)
                    if min(x2 - x1, y2 - y1) < args.min_box:
                        continue
                    # 场景图：目标物不许近乎占满整幅（那样退化成特写，测不出检测）。
                    # 但不必苛刻 —— 笔记本/闹钟在真实场景里本就占幅较大，
                    # 过严会把它们全部滤掉（实测 area_frac>0.55 时 laptop 40 张候选全废）。
                    area_frac = ((x2 - x1) * (y2 - y1)) / float(w * h)
                    if area_frac > 0.85:
                        continue
                    status = "detected"
                    box = [x1, y1, x2, y2]
                    conf = float(d.conf)

                fn = f"{key}_scene_{got:02d}.jpg"
                out_path = os.path.join(args.out, fn)
                cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                entry["images"].append({
                    "file": fn, "status": status,
                    "yolo_conf": round(conf, 3), "box": box,
                    "img_size": [w, h],
                    "area_frac": round(((box[2]-box[0])*(box[3]-box[1]))/float(w*h), 3) if box else None,
                    "src_title": c["title"],
                    "src_page": c["page"],
                    "license": c["license"],
                    "author": _first_sentence(c["author"]),
                })
                got += 1
                print(f"[{key}] {fn}  conf={conf:.2f} box={box}  {c['license']}")
        entry["n_images"] = got
        entry["n_tried"] = n_try
        # 同 key 重跑时覆盖旧条目，避免 manifest 里堆重复项
        manifest["scenes"] = [e for e in manifest["scenes"] if e.get("key") != key]
        manifest["scenes"].append(entry)
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"[{key}] done: {got} images from {n_try} candidates\n")

    total = sum(e.get("n_images", 0) for e in manifest["scenes"])
    print(f"=== 共 {total} 张场景图 → {args.out} ===")
    print(f"manifest: {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())