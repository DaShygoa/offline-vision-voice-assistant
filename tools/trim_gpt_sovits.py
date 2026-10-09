#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""裁剪 GPT-SoVITS 仓库，只保留 v3 推理链路必需的权重与源码。

背景
----
E:\\GPT-SoVITS 里预置了 v1 / v2 / v2Pro / v3 / v4 五套底模，加上训练脚本、
sv(音色相似度)、.git 等，本项目一条都用不到。本脚本按「TTS.py 实际加载路径」
反推最小集合，删除其余部分。

判定依据（不要凭感觉改，看代码）
--------------------------------
1. configs/tts_infer.yaml -> custom.version: v3
2. TTS._init_models()  -> init_t2s_weights / init_vits_weights / init_bert_weights
                          / init_cnhuhbert_weights，四处 path 都指向
                          pretrained_models/{s1v3.ckpt, s2Gv3.pth,
                          chinese-roberta-wwm-ext-large, chinese-hubert-base}
3. TTS.init_vocoder("v3") -> BigVGAN.from_pretrained(
                          pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x)
4. text/cleaner.py version != v1  ->  text/chinese2.py
5. text/chinese2.py is_g2pw = True -> text/G2PWModel (606M，不可删)
6. text/LangSegmenter/langsegmenter.py 模块级构造 LangDetector
   -> pretrained_models/fast_langdetect/lid.176.bin (126M)。
   实测：voice/tts.py:161 硬编码 text_lang="zh"，getTexts(text,"zh") 不做语种检测，
   故本项目移走该文件后构造 + 合成仍全部正常 → 非必需，可省 125M。
   仅当将来要支持 text_lang="auto" 或英文/日文时才必须保留。

用法
----
    python tools/trim_gpt_sovits.py            # dry-run，只打印清单
    python tools/trim_gpt_sovits.py --apply    # 移到 _trimmed_backup/
    python tools/trim_gpt_sovits.py --verify   # 只跑引擎自检，不删任何东西
    python tools/trim_gpt_sovits.py --apply --yes   # 跳过交互确认

删除是「移到 _trimmed_backup/」而非 rm，确认 TTS 还能合成后手动删该目录即可。
若--verify 报ModuleNotFoundError，把 _trimmed_backup 里对应项移回去，并补进 KEEP_DIRS。

实测踩过的坑（都已写进 KEEP_DIRS，勿删）
--------------------------------------
- module/models.py:13  `from f5_tts.model import DiT`—— f5_tts 看着像实验代码，其实是硬依赖
- tools/audio_sr.py:6  `sys.path.append(AP_BWE_main)` + `from datasets1.dataset import ...`
  → tools/AP_BWE_main 整棵目录树都要留
- TTS.py:33/34        tools.audio_sr / tools.i18n
- TTS.py:37           `from sv import SV` —— sv.py 与 eres2net/ 是模块级 import
- text/chinese2.py    is_g2pw = True 硬编码 → text/G2PWModel(606M) 不可删
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

# ---------------------------------------------------------------- 路径常量
DEFAULT_ROOT = r"E:\GPT-SoVITS"

# 必须保留：v3 推理链路逐个文件加载（依据见模块 docstring）
KEEP_FILES = [
    r"GPT_SoVITS\pretrained_models\s1v3.ckpt",                       # AR 语义token 预测
    r"GPT_SoVITS\pretrained_models\s2Gv3.pth",                       # VITS 主干
    r"GPT_SoVITS\pretrained_models\chinese-hubert-base",            # 语义特征提取
    r"GPT_SoVITS\pretrained_models\chinese-roberta-wwm-ext-large",  # 文本 BERT 特征
    r"GPT_SoVITS\pretrained_models\models--nvidia--bigvgan_v2_24khz_100band_256x",  # v3 声码器
    r"GPT_SoVITS\pretrained_models\fast_langdetect",# 日志分段/语种判定（实测可省，见下）
    r"nltk_data_extracted",                                          # english.py 依赖

    # ---- 训练链路：保留，因为要做「自训练音色 → 导出推理权重」----
    # 依据：s2_train_v3.py 是 v3 微调的唯一入口，
    #      utils.save_checkpoint() 产出含 optimizer 的训练断点，
    #      再由 tools/export_infer_weights.py 剥离为纯推理权重。
    r"GPT_SoVITS\s1_train.py",                       # AR 训练
    r"GPT_SoVITS\s2_train.py",                       # v2 VITS 训练
    r"GPT_SoVITS\s2_train_v3.py",                    # v3 VITS 训练（本项目目标）
    r"GPT_SoVITS\s2_train_v3_lora.py",               # v3 LoRA 训练（轻量路线）
    r"GPT_SoVITS\prepare_datasets",                  # 音频切片 + ASR + 音高/时长提取
    r"GPT_SoVITS\configs\train.yaml",                # 训练总配置
]

KEEP_DIRS = [
    r"GPT_SoVITS\text",           # 中文前端：chinese2 / G2PW / 词典 / 归一化
    r"GPT_SoVITS\configs",        # s2.json(v3 架构) + tts_infer.yaml
    r"GPT_SoVITS\AR",
    r"GPT_SoVITS\BigVGAN",
    r"GPT_SoVITS\feature_extractor",
    r"GPT_SoVITS\module",
    r"GPT_SoVITS\utils.py",
    r"GPT_SoVITS\process_ckpt.py",
    # ---- 以下都是「模块级 import」的硬依赖，删掉直接 ImportError ----
    r"GPT_SoVITS\sv.py",          # TTS.py:37from sv import SV
    r"GPT_SoVITS\eres2net",       # sv.py:7      from ERes2NetV2 import ERes2NetV2
    r"GPT_SoVITS\f5_tts",         # module/models.py:13  from f5_tts.model import DiT
    r"tools\i18n",                # TTS.py:34    from tools.i18n.i18n import I18nAuto
    r"tools\__init__.py",
    r"tools\audio_sr.py",         # TTS.py:33    from tools.audio_sr import AP_BWE
    r"tools\my_utils.py",
    r"tools\assets.py",
    r"tools\AP_BWE_main",         # audio_sr.py:6/13  sys.path + datasets1/models
]

# 可删除：其他版本底模 + 训练/多语种资产。逐条给出代码依据。
DROP_ITEMS = [
    # ---- 其他版本的底模权重，custom.version=v3 永远不会走到这些分支 ----
    (r"GPT_SoVITS\pretrained_models\gsv-v4-pretrained",
     "v4 底模，TTS.init_vocoder('v4') 才用"),
    (r"GPT_SoVITS\pretrained_models\v2Pro",
     "v2Pro/v2ProPlus 底模"),
    (r"GPT_SoVITS\pretrained_models\gsv-v2final-pretrained",
     "v2 底模，TTS_Config.default_configs['v2'] 指向它"),
    (r"GPT_SoVITS\pretrained_models\s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
     "v1 的 t2s 权重"),
    (r"GPT_SoVITS\pretrained_models\s2G488k.pth",
     "v1 的 vits 权重"),
    (r"GPT_SoVITS\pretrained_models\s2D488k.pth",
     "v1 判别器，仅训练用"),

    # ---- 重复份：text/G2PWModel 才是 chinese2.py 实际加载的 ----
    (r"GPT_SoVITS\pretrained_models\g2pw-chinese",
     "g2pw 模型的另一份拷贝，全仓库 .py 无任何引用"),

    # ---- 训练/微调链路，本项目不训练 ----
    # 注意：sv.py 与 eres2net/ 是 TTS.py 的模块级硬依赖，不能删（见 KEEP_DIRS）
    (r"GPT_SoVITS\pretrained_models\sv",
     "sv 音色相似度权重，仅 v2Pro 路径 init_sv_model() 才加载，v3 用不到"),
    (r"GPT_SoVITS\inference_webui.py", "WebUI，本项目走HTTP 自建服务"),
    (r"GPT_SoVITS\inference_webui_fast.py", "同上"),
    (r"GPT_SoVITS\inference_gui.py", "同上"),
    (r"GPT_SoVITS\onnx_export.py", "导出工具"),
    (r"GPT_SoVITS\export_torch_script.py", "导出工具"),
    (r"GPT_SoVITS\export_torch_script_v3v4.py", "导出工具"),
    (r"GPT_SoVITS\stream_v2pro.py", "v2Pro 流式推理"),
    (r"GPT_SoVITS\f5_tts", "F5-TTS 实验代码，未启用"),
    (r"GPT_SoVITS\configs\s1.yaml", "v1 训练配置"),
    (r"GPT_SoVITS\configs\s1big.yaml", "训练配置"),
    (r"GPT_SoVITS\configs\s1big2.yaml", "训练配置"),
    (r"GPT_SoVITS\configs\s1longer-v2.yaml", "训练配置"),
    (r"GPT_SoVITS\configs\s1longer.yaml", "训练配置"),
    (r"GPT_SoVITS\configs\s1mq.yaml", "训练配置"),
    (r"GPT_SoVITS\configs\s2v2Pro.json", "v2Pro 架构"),
    (r"GPT_SoVITS\configs\s2v2ProPlus.json", "v2ProPlus 架构"),

    # ---- 仓库外围 ----
    (r".git", "版本历史，本项目用不到"),
    (r".github", "CI 配置"),
    (r".pre-commit-config.yaml", "开发期钩子"),
    (r"Docker", "容器化配置"),
    (r"docs", "上游文档"),
    # 注意：tools/ 不能整删。TTS.py 依赖 tools.audio_sr / tools.i18n，
    # audio_sr.py 又依赖 tools/AP_BWE_main。tools/uvr5、asr 等才可删。
    (r"tools\asr", "训练集切分用 ASR"),
    (r"tools\denoise-model", "降噪模型"),
    (r"tools\cmd-denoise.py", "降噪脚本"),
    (r"tools\slicer2.py", "WebUI 切音工具"),
    (r"tools\subfix_webui.py", "WebUI"),
    (r"tools\uvr5", "人声分离，推理不用"),
    (r"webui.py", "WebUI 入口"),
    (r"api.py", "旧版 API 入口"),
    (r"api_v2.py", "API 入口，本项目走 tts_server.py"),
    (r"_dl_models.py", "模型下载脚本，已下载完无用"),
    (r"runtime", "Windows 运行时目录（如存在）"),
    (r"GPT_SoVITS\pretrained_models\.gitignore", "占位文件"),

    # ---- 多语种资产：只用中文 ----
    # english.py / cmudict 被 cleaner.py 的 language_module_map 引用，
    # 删掉后若误判语种会ImportError，故保留 english.py，只删日文专用词典。
    (r"GPT_SoVITS\text\ja_userdic", "日文词典，中文链路不读"),
    (r"GPT_SoVITS\text\korean.py", "韩文模块"),
    (r"GPT_SoVITS\text\cantonese.py", "粤语模块"),
]

# 这些即使在 DROP 里也别删（KEEP 优先级更高）
_PROTECTED = {os.path.normcase(p) for p in KEEP_FILES} | {
    os.path.normcase(p) for p in KEEP_DIRS
}


def dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def human(n: int) -> str:
    for unit in ("B", "K", "M", "G"):
        if n < 1024 or unit == "G":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}G"


def collect(root: str):
    """返回 (存在的可删项, 缺失项, 释放字节数)"""
    found, missing, freed = [], [], 0
    for rel, why in DROP_ITEMS:
        if os.path.normcase(rel) in _PROTECTED:
            continue
        full = os.path.join(root, rel)
        if not os.path.exists(full):
            missing.append(rel)
            continue
        size = dir_size(full) if os.path.isdir(full) else os.path.getsize(full)
        found.append((rel, why, size))
        freed += size
    return found, missing, freed


def verify_keep(root: str) -> list:
    """校验所有 KEEP 项都在，否则说明仓库结构变了，不能删。"""
    return [p for p in KEEP_FILES + KEEP_DIRS if not os.path.exists(os.path.join(root, p))]


def collect_imports(root: str) -> set:
    """扫描保留源码里的顶层 import，得出「本地模块名」集合。"""
    import re
    names = set()
    pat = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][\w]*)", re.M)
    for rel_dir in (r"GPT_SoVITS", r"tools"):
        base = os.path.join(root, rel_dir)
        for dirpath, _dirs, files in os.walk(base):
            for f in files:
                if not f.endswith(".py"):
                    continue
                try:
                    with open(os.path.join(dirpath, f), "r", encoding="utf-8",
                              errors="ignore") as fh:
                        names.update(pat.findall(fh.read()))
                except OSError:
                    pass
    return names


def local_module_index(root: str) -> set:
    """仓库内可被 import 的本地模块/包名（含 sys.path.append 注入的目录）。"""
    mods = set()
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs
                   if d not in {"__pycache__", "_trimmed_backup", ".git"}]
        for d in dirs:
            if os.path.exists(os.path.join(dirpath, d, "__init__.py")):
                rel = os.path.relpath(os.path.join(dirpath, d), root)
                mods.add(rel.replace("\\", "."))
        for f in files:
            if f.endswith(".py"):
                rel = os.path.relpath(os.path.join(dirpath, f), root)
                mods.add(rel[:-3].replace("\\", "."))
    # audio_sr.py 把 AP_BWE_main 塞进 sys.path，里面的包也算本地模块
    for extra in ("datasets1", "models"):
        if os.path.isdir(os.path.join(root, "tools", "AP_BWE_main", extra)):
            mods.add(extra)
    return mods


def check_dangling_imports(root: str, dropped: set) -> list:
    """删除后会不会有 import 悬空。返回 [(模块, 引用方)] 。"""
    local = local_module_index(root)
    imported = collect_imports(root)
    return sorted(m for m in imported & local if m in dropped)


def verify_engine(root: str) -> int:
    """实测：裁剪后能否加载引擎并合成一句话。

    必须用 GPTSoVits 环境跑（本项目 vision_ai 里没有 torchaudio）。
    这是唯一可信的验证方式 —— 静态 import 扫描抓不到 sys.path.append
    和字符串拼路径造成的隐式依赖。
    """
    import subprocess
    import textwrap
    py = r"C:\Users\admin\anaconda3\envs\GPTSoVits\python.exe"
    if not os.path.isfile(py):
        print(f"[x] 找不到 {py}")
        return 2
    code = textwrap.dedent(f"""
        import os, sys, numpy as np
        os.chdir({root!r})
        for p in ({root!r}, os.path.join({root!r}, "GPT_SoVITS")):
            if p not in sys.path:
                sys.path.insert(0, p)
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
        c = TTS_Config(); c.device = "cpu"; c.is_half = False
        c.vits_weights_path = os.path.join({root!r}, "GPT_SoVITS", "pretrained_models", "s2Gv3.pth")
        c.t2s_weights_path = os.path.join({root!r}, "GPT_SoVITS", "pretrained_models", "s1v3.ckpt")
        t = TTS(c)
        # TTS.run() 是 generator，产出 (sr, ndarray)
        sr, a = list(t.run({{
            "text": "\u8fd9\u662f\u4f60\u7684\u9a6c\u514b\u676f",
            "text_lang": "zh",
            "ref_audio_path": r"E:\\cv\\models\\gpt_sovits\\ref_audio.wav",
            "prompt_text": "\u5bf9\u6211\u505a\u4e86\u4ecb\u7ecd\u554a\u90a3\u4e48\u6211\u60f3\u8bf4\u7684\u8bdd\u662f\u5462\u5927\u5bb6\u5982\u679c\u5bf9\u6211\u7684\u7814\u7a76\u611f\u5174\u8da3\u5462\u4f60",
            "prompt_lang": "zh",
            "text_split_method": "cut0", "top_k": 15, "speed_factor": 1.0,
        }}))[0]
        a = np.asarray(a)
        print("VERIFY_OK sr=%d dur=%.2fs peak=%.0f"
              % (sr, a.size / sr, float(np.max(np.abs(a)))))
    """)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    proc = subprocess.run([py, "-c", code], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env, timeout=600)
    tail = [ln for ln in (proc.stdout or "").splitlines()
            if "VERIFY_OK" in ln or "Error" in ln or "error" in ln]
    for ln in tail[-6:]:
        print("  " + ln)
    if proc.returncode != 0 or not any("VERIFY_OK" in ln for ln in tail):
        print("\n[x] 验证失败。最后 15 行输出：")
        for ln in (proc.stderr or proc.stdout or "").splitlines()[-15:]:
            print("      " + ln)
        print("\n把 _trimmed_backup 里缺的项移回原位，再补进脚本 KEEP_DIRS。")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="裁剪 GPT-SoVITS 到 v3 推理最小集")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="GPT-SoVITS 仓库根目录")
    ap.add_argument("--apply", action="store_true", help="执行裁剪（移到备份目录）")
    ap.add_argument("--yes", action="store_true", help="跳过交互确认")
    ap.add_argument("--verify", action="store_true",
                    help="只跑引擎自检（加载 + 合成一句话），不删任何东西")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"[x] 仓库不存在: {root}")
        return 2

    if args.verify:
        print(f"验证 {root} 的 v3 推理链路 ...")
        return verify_engine(root)

    missing_keep = verify_keep(root)
    if missing_keep:
        print("[x] 以下必需项缺失，仓库结构可能已变化，拒绝执行：")
        for p in missing_keep:
            print(f"      - {p}")
        return 2

    before = dir_size(root)
    found, missing, freed = collect(root)

    # 悬空 import 自检：防止删掉仍被模块级 import 的东西
    dropped_top = {os.path.splitext(os.path.basename(rel))[0]
                   for rel, _why, _size in found}
    dangling = check_dangling_imports(root, dropped_top)
    if dangling:
        print("[!] 警告：以下模块仍被 import，删除后会 ImportError：")
        for m in dangling:
            print(f"      - {m}")
        print()

    print(f"仓库: {root}")
    print(f"当前: {human(before)}")
    print(f"可删: {len(found)} 项，释放 {human(freed)}")
    print(f"裁剪后:约 {human(before - freed)}")
    print()
    for rel, why, size in found:
        print(f"  {human(size):>7}  {rel}")
        print(f"           {why}")
    if missing:
        print(f"\n已不存在（跳过 {len(missing)} 项）: {', '.join(missing[:5])}"
              f"{' ...' if len(missing) > 5 else ''}")

    if not args.apply:
        print("\n[dry-run] 加 --apply 执行实际删除。")
        return 0

    if not args.yes:
        ans = input(f"\n确认删除 {len(found)} 项、释放 {human(freed)}？输入 yes 继续: ")
        if ans.strip().lower() != "yes":
            print("已取消。")
            return 1

    #先移到回收目录再彻底删，出错还能救回来
    trash = os.path.join(root, "_trimmed_backup")
    os.makedirs(trash, exist_ok=True)
    ok = fail = 0
    for rel, _why, _size in found:
        src = os.path.join(root, rel)
        dst = os.path.join(trash, rel.replace("\\", "__").replace("/", "__"))
        try:
            if os.path.isdir(src):
                shutil.move(src, dst)
            else:
                shutil.move(src, dst)
            ok += 1
        except Exception as exc:                # noqa: BLE001
            print(f"[!] 移动失败 {rel}: {exc}")
            fail += 1

    print(f"\n已移入 {trash}: {ok} 项，失败 {fail} 项")
    print("先跑一次 TTS 合成验证；确认无误后手动删除 _trimmed_backup 即可。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
