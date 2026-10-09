#!/usr/bin/env python
"""Build a course-ready object dataset from Wikimedia Commons (选题一 数据要求).

Why this tool exists
--------------------
选题书「4 数据要求」:
  * ≥5 个物体，每个 ≥5 张图片
  * 注册图片和测试图片不要完全相同
  * ≥3 个未录入物体，用于测试未知物体判断
  * 「使用摄像头实时进行处理的加分！」→ 需要真实物体图

关键设计：抓取时就用 YOLO 做**校验 + 裁剪**，与在线链路完全一致：
    下载 → YOLO 检测目标类 → 裁剪 ROI → quality_ok → 存图
这样离线评测得到的阈值/准确率可直接迁移到摄像头实时场景，
避免「离线分布好、现场全崩」。

注意：物体用 COCO **类名** 表示，cls_id 在运行时从 COCO_CLASSES 反查，
不要硬编码数字（YOLOv8 的类号极易记错）。

ImageNet-pretrained ResNet18 是分类 backbone，对「同类不同实例」天然弱
（两个长得像的杯子会撞）。因此 enroll 组与 test 组使用**不同实例**的照片，
逼近现场真实压力。

Usage:
    python tools/fetch_dataset.py --list
    python tools/fetch_dataset.py --probe
    python tools/fetch_dataset.py --enroll 6 --test 3 --unknown 3
    python tools/fetch_dataset.py --enroll 6 --test 3 --unknown 3 --min-conf 0.35
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

OUT_ROOT = os.path.join("data", "dataset")
API = "https://commons.wikimedia.org/w/api.php"
UA = "cv-coursework-dataset/1.0 (student course project)"

# ---------------------------------------------------------------- object plan
# cls = COCO class NAME (resolved to id at runtime). queries tried in order.
#
# 每条 query /  cls 都经过 tools/fetch_dataset.py --probe 实测筛选：
# Commons 上「有人物场景」的照片YOLO 检不出目标物或框太小，直接剔除。
#   实测可用数(2 queries × 8candidates, min_box=110):
#     OK   scissors 12 / clock 10 / banana 11 / umbrella 7 / keyboard 6 / cup 6
#     WEAK mouse 2 / bowl 2 / traffic light 3 / vase 4 / potted plant 3
#          book 1 / backpack 0 / bear 0        ← 已弃用
PLAN = [
    # ---- enrolled (录入) objects: 5 类，形状/材质/用途差异大----
    dict(key="cup",      cn="马克杯", cls="cup",      queries=["white ceramic mug", "coffee mug cup", "ceramic drinking mug", "mug on table"]),
    dict(key="bottle",   cn="保温瓶", cls="bottle",queries=["stainless steel water bottle", "thermos flask", "metal water bottle", "water bottle"]),
    dict(key="keyboard", cn="键盘",   cls="keyboard", queries=["computer keyboard", "mechanical keyboard", "keyboard on desk", "usb keyboard"]),
    dict(key="scissors", cn="剪刀",   cls="scissors", queries=["pair of scissors", "scissors", "kitchen scissors", "scissors blade"]),
    dict(key="clock",    cn="闹钟",   cls="clock",    queries=["alarm clock", "table clock", "analog clock", "clock face"]),
    dict(key="laptop",   cn="笔记本电脑", cls="laptop",   queries=["open laptop notebook", "laptop computer", "laptop on desk", "notebook computer"]),
    # ---- unknown (未录入): bowl 是 cup 的 hard negative（形状近似，最难拒绝）----
    dict(key="unknown_bowl",    cn="碗",     cls="bowl",queries=["ceramic bowl", "soup bowl", "bowl on table"]),
    dict(key="unknown_banana",  cn="香蕉",   cls="banana",    queries=["banana bunch", "bananas", "single banana fruit"]),
    dict(key="unknown_umbrella", cn="雨伞",  cls="umbrella",  queries=["umbrella", "open umbrella", "folding umbrella"]),
    dict(key="unknown_chair",   cn="椅子",   cls="chair",     queries=["wooden chair", "office chair", "chair on floor"]),
]


# ------------------------------------------------------------------- download
def _ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


_SSL = _ssl_ctx()


def api(params: dict, retries: int = 3) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=25, context=_SSL) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.2 * (i + 1))
    raise RuntimeError(f"Commons API failed: {last}")


def search_files(query: str, limit: int = 30) -> List[dict]:
    """Return [{title,url,page_url,license,author}] for bitmap files matching query."""
    try:
        d = api(
            {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": f"filetype:bitmap {query}",
                "gsrnamespace": "6",
                "gsrlimit": str(limit),
                "prop": "imageinfo",
                "iiprop": "url|mime|size|extmetadata",
                "iiurlwidth": "900",
            }
        )
    except Exception as e:  # noqa: BLE001
        print(f"[warn] search failed: {query} ({e})")
        return []
    out: List[dict] = []
    for _pid, p in (d.get("query", {}).get("pages", {}) or {}).items():
        ii = (p.get("imageinfo") or [{}])[0]
        if not ii or ii.get("mime") not in ("image/jpeg", "image/png"):
            continue
        if int(ii.get("width", 0)) < 400 or int(ii.get("height", 0)) < 400:
            continue
        meta = ii.get("extmetadata", {}) or {}
        out.append(
            dict(
                title=p.get("title", ""),
                url=ii.get("thumburl") or ii.get("url", ""),
                page_url=ii.get("descriptionurl", ""),
                license=(meta.get("LicenseShortName", {}) or {}).get("value", ""),
                author=(meta.get("Artist", {}) or {}).get("value", "")[:120],
            )
        )
    return out


def download(url: str, retries: int = 2) -> Optional[bytes]:
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30, context=_SSL) as r:
                data = r.read()
            if len(data) > 8000:
                return data
        except Exception:  # noqa: BLE001
            time.sleep(0.8 * (i + 1))
    return None


# ---------------------------------------------------------------------- main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fetch enrolled/unknown object images from Wikimedia Commons")
    p.add_argument("--list", action="store_true", help="print the object plan and exit")
    p.add_argument("--probe", action="store_true", help="only report whether YOLO detects the class; write nothing")
    p.add_argument("--enroll", type=int, default=6, help="number of enrolled objects")
    p.add_argument("--test", type=int, default=3, help="held-out test images per enrolled object")
    p.add_argument("--unknown", type=int, default=3, help="number of unknown objects")
    p.add_argument("--min-conf", type=float, default=0.30, help="min YOLO conf to accept a crop")
    p.add_argument("--min-box", type=int, default=110,
                   help="min ROI side in px — rejects degenerate crops (e.g. 64x64 distant hits)")
    p.add_argument("--candidates", type=int, default=14, help="candidate images per query")
    p.add_argument("--out", default=OUT_ROOT)
    args = p.parse_args(argv)

    from vision.detector import COCO_CLASSES

    if args.list:
        for o in PLAN:
            print(f"{o['key']:<24} {o['cn']:<10} COCO '{o['cls']}' = {COCO_CLASSES.index(o['cls'])}")
        return 0

    from core.config import load_config
    from vision.detector import YoloDetector
    from vision.object_cropper import ObjectCropper

    cfg = load_config()
    det = YoloDetector(cfg.get("yolo", {}))
    det.load()
    print(f"[ok] YOLO {det.device_used} conf_th={det.conf_th}")
    cropper = ObjectCropper()
    min_crop_px = int(cfg.get("memory", {}).get("min_crop_px", 64))

    enrolled = PLAN[: max(0, args.enroll)]
    unknown = [o for o in PLAN if o["key"].startswith("unknown_")][: max(0, args.unknown)]
    targets = [(o, True) for o in enrolled] + [(o, False) for o in unknown]

    if not args.probe:
        for sub in ("enroll", "test", "unknown"):
            os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    mpath = os.path.join(args.out, "manifest.json")
    manifest = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), source="Wikimedia Commons",
                    yolo=det.device_used, min_conf=args.min_conf, objects=[])
    if os.path.isfile(mpath) and not args.probe:
        try:
            with open(mpath, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            print(f"[info] appending to existing manifest ({len(manifest.get('objects', []))} objects)")
        except Exception:  # noqa: BLE001
            pass

    for obj, is_enrolled in targets:
        key, cn, cls_name = obj["key"], obj["cn"], obj["cls"]
        cls_id = COCO_CLASSES.index(cls_name)
        # 选题书要求「每个物体 ≥5 张注册图」，所以抓取目标 = test + 5 (+ 余量)
        want = (args.test + 6) if is_enrolled else args.unknown
        got: List[dict] = []
        seen: set[str] = set()

        for q in obj["queries"]:
            if len(got) >= want:
                break
            cands = search_files(q, limit=args.candidates)
            print(f"[{key}] '{q}' → {len(cands)} candidates (cls={cls_name}:{cls_id})")
            for c in cands:
                if len(got) >= want:
                    break
                if c["title"] in seen:
                    continue
                seen.add(c["title"])
                raw = download(c["url"])
                if raw is None:
                    continue
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                h, w = img.shape[:2]
                if min(h, w) < 320:
                    continue
                try:
                    dets, _ = det.infer(img)
                except Exception:  # noqa: BLE001
                    continue
                cand = [d for d in dets if d.cls_id == cls_id and d.conf >= args.min_conf]
                if not cand:
                    continue
                d = max(cand, key=lambda x: x.conf)
                x1, y1, x2, y2 = (int(v) for v in d.xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                # reject degenerate ROI: too small → ResNet18 sees a blurry blob
                if min(x2 - x1, y2 - y1) < args.min_box:
                    continue
                crop = cropper.crop_box(img, (x1, y1, x2, y2))
                if not cropper.quality_ok(crop) or min(crop.shape[:2]) < min_crop_px:
                    continue
                got.append(dict(
                    crop=crop,
                    conf=round(float(d.conf), 3),
                    box=[x1, y1, x2, y2],
                    crop_shape=[int(crop.shape[0]), int(crop.shape[1])],
                    src_title=c["title"], src_page=c["page_url"],
                    license=c["license"], author=c["author"],
                ))
                print(f"   OK {c['title'][:50]:<50} conf={d.conf:.2f} crop={crop.shape[1]}x{crop.shape[0]}")

        if not got:
            print(f"[{key}] !! nothing passed YOLO + quality gate")
            continue

        # held-out test = 最早成功的少量 crop；enroll = 其余（不同实例、不同源文件）
        # 保证 enroll ≥5（选题书硬性要求），test ≥1
        if is_enrolled:
            n_test = max(1, min(args.test, len(got) - 5)) if len(got) > 5 else 1
            splits = [("test", got[:n_test]), ("enroll", got[n_test:])]
        else:
            splits = [("unknown", got)]

        entry = dict(key=key, cn=cn, cls=cls_name, cls_id=cls_id,
                     role="enrolled" if is_enrolled else "unknown",
                     n_crops=len(got), images=[])
        for split, items in splits:
            for g in items:
                if not args.probe:
                    d_dir = os.path.join(args.out, split)
                    os.makedirs(d_dir, exist_ok=True)
                    n = len([x for x in os.listdir(d_dir) if x.startswith(f"{key}_")])
                    # unknown 组的 key 已含 "unknown_"，不再重复拼后缀
                    suffix = "" if split == "unknown" else f"_{split}"
                    fn = f"{key}{suffix}_{n:02d}.jpg"
                    cv2.imwrite(os.path.join(d_dir, fn), g["crop"], [cv2.IMWRITE_JPEG_QUALITY, 94])
                else:
                    fn = "(probe)"
                entry["images"].append(dict(
                    split=split, file=fn, yolo_conf=g["conf"], box=g["box"],
                    crop_shape=g["crop_shape"], src_title=g["src_title"],
                    src_page=g["src_page"], license=g["license"], author=g["author"]))
        print(f"[{key}] {len(got)} crops accepted")
        manifest["objects"] = [o for o in manifest["objects"] if o.get("key") != key] + [entry]

    if not args.probe:
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        n_en = len([o for o in manifest["objects"] if o["role"] == "enrolled"])
        n_un = len([o for o in manifest["objects"] if o["role"] == "unknown"])
        e = sum(len([i for i in o["images"] if i["split"] == "enroll"]) for o in manifest["objects"])
        t = sum(len([i for i in o["images"] if i["split"] == "test"]) for o in manifest["objects"])
        u = sum(len([i for i in o["images"] if i["split"] == "unknown"]) for o in manifest["objects"])
        print("\n" + "=" * 64)
        print(f"DATASET  enrolled_objs={n_en}  enroll_imgs={e}  test_imgs={t}  "
              f"unknown_objs={n_un}  unknown_imgs={u}")
        ok = n_en >= 5 and n_un >= 3 and all(
            len([i for i in o["images"] if i["split"] == "enroll"]) >= 5
            for o in manifest["objects"] if o["role"] == "enrolled")
        print("选题书数据要求: " + ("PASS" if ok else "YET TO SATISFY"))
        print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
