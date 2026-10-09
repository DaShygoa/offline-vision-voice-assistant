# -*- coding: utf-8 -*-
"""把 GPT-SoVITS 训练产出的 checkpoint 导出成「纯推理权重」。

为什么需要
----------
`s2_train_v3.py` 的 `utils.save_checkpoint()` 保存的是**训练断点**：
    {"model": state_dict, "iteration": n, "optimizer": optim.state_dict(), "learning_rate": lr}
其中 optimizer 状态通常和模型同量级甚至更大。而 `pretrained_models/s2Gv3.pth`
这种官方底模只有：
    {"weight": state_dict, "config": hps, "info": "pretrained_s2G_v3"}
推理只需要 weight + config，多余的 optimizer 直接扔掉即可瘦身。

另外还做三件事（都与官方 process_ckpt.savee() 对齐）：
  1. config 继承：训练断点**没有** config 键（save_checkpoint 只存 model/optimizer），
     但 TTS.py:506 `hps = dict_s2["config"]` 是无条件读取 —— 缺了就 KeyError。
     故从对应版本的官方底模继承一份（架构与底模一致时这是等价的）。
  2. 剔除 enc_q：官方 savee() 显式 `if "enc_q" in key: continue`，
     那是判别器分支，推理用不到。
  3. fp32 -> fp16：官方 savee() 里 `opt["weight"][key] = ckpt[key].half()` 是硬编码的，
     实测官方底模 s2Gv3.pth 的 685 个张量**全部**是 float16。故默认转 fp16。

用法
----
    python tools/export_infer_weights.py --src <G_xxx.pth> --dst s2G_mine.pth
    python tools/export_infer_weights.py --list-logs          # 列出可导出的训练断点
    python tools/export_infer_weights.py --src ... --dst ... --fp32
"""
from __future__ import annotations

import argparse
import os
import sys

# 推理必需的键（对照 load_sovits_new + TTS.init_vits_weights）
NEEDED = ("weight", "config")

# config 继承来源：与训练版本对应的官方底模。
# 训练断点没有 config，但 TTS.py:506 无条件读 dict_s2["config"]，必须补上。
# 架构一致时（v3 训练 v3 底模）这份 config 就是正确的。
CONFIG_SOURCE = {
    "v1": r"GPT_SoVITS\pretrained_models\s2G488k.pth",
    "v2": r"GPT_SoVITS\pretrained_models\gsv-v2final-pretrained\s2G.pth",
    "v3": r"GPT_SoVITS\pretrained_models\s2Gv3.pth",
    "v4": r"GPT_SoVITS\pretrained_models\gsv-v4-pretrained\s2G.pth",
}


def human(n: float) -> str:
    for u in ("B", "K", "M", "G"):
        if n < 1024 or u == "G":
            return "%.1f%s" % (n, u) if u != "B" else "%dB" % n
        n /= 1024.0
    return "%.1fG" % n


def file_size(p: str) -> int:
    try:
        return os.path.getsize(p)
    except OSError:
        return 0


def list_logs(root: str) -> int:
    """扫出所有训练断点，标出可瘦身空间。"""
    if not os.path.isdir(root):
        print("[x] 目录不存在: %s" % root)
        return 2
    rows = []
    for dirpath, _d, files in os.walk(root):
        for f in files:
            if not f.endswith(".pth"):
                continue
            full = os.path.join(dirpath, f)
            rows.append((full, file_size(full)))
    if not rows:
        print("没找到 .pth 训练断点。先跑 s2_train_v3.py 训练。")
        return 1
    rows.sort(key=lambda x: -x[1])
    print("共 %d 个 .pth：" % len(rows))
    for full, size in rows[:40]:
        rel = os.path.relpath(full, root)
        # 估算：optimizer 大约占一半
        print("  %8s  %s" % (human(size), rel))
    print("\n提示：官方底模 s2Gv3.pth 为 734M，而训练断点通常 1.4G 左右，"
          "差值就是 optimizer 状态。")
    return 0


def _guess_version(weight: dict, file_bytes: int) -> str:
    """按官方 get_sovits_version_from_path_fast 的同一套规则判定版本。

    注意别踩这个坑：TTS.py:508-511 里`shape[0] == 322` 判的是 **v1**
    （源码注释写的是 "v3model,v2symbols"），v3 不走这个分支。
    v3 的判定靠文件体积：>= 700MB 视为 v3（process_ckpt.py:76-80）。
    这里沿用体积规则，因为 v3 训练断点必然 > 1G。
    """
    if "enc_p.text_embedding.weight" not in weight:
        return "v2"                      # 缺 text_embedding → v3/v4 系（官方判v2 符号集）
    if int(weight["enc_p.text_embedding.weight"].shape[0]) == 322:
        return "v1"
    if file_bytes >= 700 * 1024 * 1024:
        return "v3"
    return "v2"


def load_config_template(version: str) -> dict | None:
    import torch
    src = CONFIG_SOURCE.get(version)
    if not src:
        return None
    full = os.path.join(r"E:\GPT-SoVITS", src)
    if not os.path.isfile(full):
        print("[!] 找不到 config 模板底模: %s" % full)
        return None
    print("从官方底模继承 config: %s" % src)
    d = torch.load(full, map_location="cpu", weights_only=False)
    return d.get("config")


def export(src: str, dst: str, to_fp16: bool = True,
           drop_optimizer: bool = True) -> int:
    import torch

    if not os.path.isfile(src):
        print("[x] 源文件不存在: %s" % src)
        return 2

    print("读取 %s (%s)" % (src, human(file_size(src))))
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict):
        print("[x] 不是 dict，无法导出")
        return 2

    print("原始键: %s" % list(ckpt.keys()))

    # 训练断点的权重键叫 "model"，官方底模叫 "weight"
    weight = ckpt.get("weight")
    key_used = "weight"
    if weight is None and "model" in ckpt:
        weight = ckpt["model"]
        key_used = "model"
    if weight is None:
        print("[x] 既没有 weight 也没有 model 键，放弃")
        return 2

    version = _guess_version(weight, file_size(src))
    print("识别为 %s" % version)

    # config：训练断点没有，必须从官方底模继承（见模块 docstring）
    cfg = ckpt.get("config")
    if cfg is None:
        print("[!] 该断点没有 config 键（save_checkpoint 本来就不存）。")
        cfg = load_config_template(version)
        if cfg is None:
            print("[x] 无法取得 config，无法导出。"
                  "请手动从官方底模里取config 后用 --config-from 指定。")
            return 2

    # 剔除 enc_q —— 对齐官方 savee()："if enc_q in key: continue"
    before_n = len(weight)
    weight = {k: v for k, v in weight.items() if "enc_q" not in k}
    if len(weight) != before_n:
        print("剔除 enc_q %d 个张量（判别器分支，推理不用）"
              % (before_n - len(weight)))

    if to_fp16:
        weight = {k: (v.half() if v.is_floating_point() else v)
                  for k, v in weight.items()}
        print("已转 fp16（官方 savee() 就是 .half() 硬编码，底模实测亦全 fp16）")
    else:
        print("保持原 dtype")

    out = {"weight": weight, "config": cfg,
           "info": ckpt.get("info", "%sepoch_exported" % version)}
    if "lora_rank" in ckpt:                 # LoRA 权重必须带上
        out["lora_rank"] = ckpt["lora_rank"]
        print("检测到 lora_rank = %s（LoRA 权重，保留）" % ckpt["lora_rank"])

    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    torch.save(out, dst)
    before, after = file_size(src), file_size(dst)
    print("\n导出完成")
    print("  源  %s  (%s)" % (src, human(before)))
    print("  目标 %s (%s)" % (dst, human(after)))
    if before:
        print("  瘦身 %.1f%%" % (100.0 * (1 - after / float(before))))
    print("  weight 张量数 %d（来自 '%s' 键）" % (len(weight), key_used))

    # 回读校验：确认能被 load_sovits_new 读出
    from GPT_SoVITS.process_ckpt import load_sovits_new  # noqa
    print("\n回读校验...")
    d2 = load_sovits_new(dst)
    missing = [k for k in NEEDED if k not in d2]
    if missing:
        print("[x] 校验失败，缺少键: %s" % missing)
        return 1
    print("[OK] load_sovits_new 可读，含 weight + config")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 GPT-SoVITS 纯推理权重")
    ap.add_argument("--src", help="训练断点 G_xxx.pth")
    ap.add_argument("--dst", help="导出的推理权重路径")
    ap.add_argument("--fp32", action="store_true",
                    help="保持原 dtype 不转 fp16（默认 fp16，与官方底模一致）")
    ap.add_argument("--list-logs", action="store_true", help="列出可导出的断点")
    ap.add_argument("--logs-root",
                    default=r"E:\GPT-SoVITS\GPT_SoVITS\logs_s2_v3",
                    help="训练断点根目录")
    args = ap.parse_args()

    if os.getcwd().lower() != r"e:\gpt-sovits".lower():
        os.chdir(r"E:\GPT-SoVITS")
    for p in (r"E:\GPT-SoVITS", r"E:\GPT-SoVITS\GPT_SoVITS"):
        if p not in sys.path:
            sys.path.insert(0, p)

    if args.list_logs:
        return list_logs(args.logs_root)
    if not args.src or not args.dst:
        ap.print_help()
        return 2
    return export(args.src, args.dst, to_fp16=not args.fp32)


if __name__ == "__main__":
    sys.exit(main())