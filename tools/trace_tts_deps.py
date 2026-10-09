"""追踪 GPT-SoVITS 推理真实读取的仓库文件（权威依赖清单）。

用 sys.addaudithook 监听 open 事件，只记录 E:\\GPT-SoVITS 内的真实磁盘文件。
比静态 import 扫描可靠：能看到 torch.load 实际读了哪些权重、
transformers 读了哪些分词器文件、TextPreprocessor 读了哪些词典。
"""
import os
import sys

ROOT = r"E:\GPT-SoVITS"
os.chdir(ROOT)
for _p in (ROOT, os.path.join(ROOT, "GPT_SoVITS")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_NORM_ROOT = os.path.normcase(ROOT)
_opened = {}


def _hook(event, args):
    if event != "open":
        return
    try:
        f = args[0]
        if isinstance(f, bytes):
            f = f.decode("utf-8", "ignore")
        if not isinstance(f, str):
            return
        # 用 normcase 比较（Windows 不区分大小写），但保留原始大小写用于展示
        fn = os.path.abspath(f)
        if os.path.normcase(fn).startswith(_NORM_ROOT):
            _opened[os.path.normcase(fn)] = fn
    except Exception:
        pass


sys.addaudithook(_hook)

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

cfg = TTS_Config()
cfg.device = "cpu"
cfg.is_half = False
cfg.vits_weights_path = os.path.join(ROOT, "GPT_SoVITS", "pretrained_models", "s2Gv3.pth")
cfg.t2s_weights_path = os.path.join(ROOT, "GPT_SoVITS", "pretrained_models", "s1v3.ckpt")

tts = TTS(cfg)
sr, audio = list(tts.run({
    "text": "这是你的马克杯",
    "text_lang": "zh",
    "ref_audio_path": r"E:\cv\models\gpt_sovits\ref_audio.wav",
    "prompt_text": "对我做了介绍啊那么我想说的话是呢大家如果对我的研究感兴趣呢你",
    "prompt_lang": "zh",
    "text_split_method": "cut0",
    "top_k": 15,
    "speed_factor": 1.0,
}))[0]

# 只保留磁盘上真实存在的文件
real = {}
for fn in _opened.values():
    if not os.path.isfile(fn):
        continue
    try:
        size = os.path.getsize(fn)
    except OSError:
        size = 0
    real[os.path.relpath(fn, ROOT).replace("\\", "/")] = size


def group_of(rel):
    low = rel.lower()
    if "/pretrained_models/" in low:
        return "pretrained_models/" + rel.split("/")[-2]
    if "/text/g2pwm" in low:                      # text/G2PWModel
        return "text/G2PWModel"
    if "/text/" in low:
        return "text/<前端与词典>"
    if low.startswith("gpt_sovits/"):
        return "GPT_SoVITS 源码:" + rel.split("/")[-2 if rel.count("/") > 1 else -1]
    if low.startswith("tools/"):
        return "tools/ 源码"
    return rel.split("/")[0] + "/"


groups = {}
for rel, size in real.items():
    groups[group_of(rel)] = groups.get(group_of(rel), 0) + size

print("=" * 64)
print("Python 层 open 捕获: %d 个文件, 合计 %.0f MB   (合成 %d 样本 @ %d Hz)"
      % (len(real), sum(real.values()) / 1048576, len(audio), sr))
print("=" * 64)
for g, size in sorted(groups.items(), key=lambda kv: -kv[1]):
    print("%9.1f MB  %s" % (size / 1048576, g))

print("-" * 64)
print("【权重/模型文件明细 (>1MB)】")
for rel, size in sorted(real.items(), key=lambda kv: -kv[1]):
    if size > 1048576:
        print("%9.1f MB  %s" % (size / 1048576, rel))

print("-" * 64)
cfg_files = [r for r in real if "/configs/" in r.lower()]
print("【是否读 configs/】")
print("  ", ", ".join(cfg_files) if cfg_files else "未读取 —— 架构内嵌在 .pth 的 config 字段")

nltk = [r for r in real if "nltk" in r.lower()]
print("【是否读 nltk】")
print("  ", ", ".join(nltk[:3]) if nltk else "未读取（本次全中文，未走 english 分支）")

print("-" * 64)
print("【text/ 下的词典文件 (<1MB)】")
for rel, size in sorted(real.items(), key=lambda kv: -kv[1]):
    if "/text/" in rel.lower() and "g2pwm" not in rel.lower():
        print("%9.3f MB  %s" % (size / 1048576, rel))
