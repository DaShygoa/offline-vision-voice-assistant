"""非破坏性依赖验证：模拟「某文件/模块不存在」，看引擎是否还能跑。

比移走文件好在
1. 不碰文件系统，零风险，不会被安全网关拦；
2. 可以精确模拟「文件不存在」（os.path.exists 返回 False、open 抛 FileNotFoundError），
   而移走文件时onnxruntime / mmap 可能仍从已打开句柄读到内容。

原理
----
- 屏蔽模块：在 sys.meta_path 前插一个 finder，指定名字直接抛 ModuleNotFoundError，
  等价于「源码+pyc 都不存在」。
- 屏蔽文件：包一层 os.path.exists / builtins.open，命中目标路径就报 FileNotFoundError。
  对 onnxruntime 这类C++ mmap 加载的场景，屏蔽 os.path.exists 就能触发它自己的报错。

用法
----
    python tools/probe_tts_deps_nondestructive.py
"""
import os
import subprocess
import sys
import tempfile

ROOT = r"E:\GPT-SoVITS"
PY = r"C:\Users\admin\anaconda3\envs\GPTSoVits\python.exe"

# (标签, 类型, 目标)   type: "module" | "file"
CASES = [
    ("GPT_SoVITS/sv.py",              "module", "sv"),
    ("GPT_SoVITS/eres2net/",          "module", "ERes2NetV2"),
    ("GPT_SoVITS/f5_tts/",            "module", "f5_tts.model"),
    ("tools/AP_BWE_main/",            "module", "datasets1.dataset"),
    ("tools/audio_sr.py",             "module", "tools.audio_sr"),
    ("tools/i18n/",                   "module", "tools.i18n.i18n"),
    ("text/G2PWModel/g2pW.onnx 606M", "file",
     r"E:\GPT-SoVITS\GPT_SoVITS\text\G2PWModel\g2pW.onnx"),
    ("text/g2pw/polyphonic.pickle",   "file",
     r"E:\GPT-SoVITS\GPT_SoVITS\text\g2pw\polyphonic.pickle"),
    ("text/opencpop-strict.txt",      "file",
     r"E:\GPT-SoVITS\GPT_SoVITS\text\opencpop-strict.txt"),
    ("text/zh_normalization/",        "file",
     r"E:\GPT-SoVITS\GPT_SoVITS\text\zh_normalization\text_normlization.py"),
    ("configs/tts_infer.yaml",        "file",
     r"E:\GPT-SoVITS\GPT_SoVITS\configs\tts_infer.yaml"),
    ("configs/s2.json",               "file",
     r"E:\GPT-SoVITS\GPT_SoVITS\configs\s2.json"),
    ("pretrained_models/fast_langdetect/", "file",
     r"E:\GPT-SoVITS\GPT_SoVITS\pretrained_models\fast_langdetect\lid.176.bin"),
    ("pretrained_models/sv/",         "file",
     r"E:\GPT-SoVITS\GPT_SoVITS\pretrained_models\sv\pretrained_eres2netv2w24s4ep4.ckpt"),
]

TEMPLATE = r'''
import os, sys, builtins, importlib.abc, importlib.machinery

BLOCK_MODULE = {block_module!r}
BLOCK_FILE = {block_file!r}

# ---- ① 屏蔽文件：包一层 os.path.exists / open ----
if BLOCK_FILE:
    _real_exists = os.path.exists
    _real_open = builtins.open
    _target = os.path.normcase(os.path.abspath(BLOCK_FILE))

    def _fake_exists(path):
        try:
            ap = os.path.normcase(os.path.abspath(os.fspath(path)))
        except Exception:
            return _real_exists(path)
        return False if ap == _target else _real_exists(path)

    def _fake_open(file, *a, **kw):
        try:
            ap = os.path.normcase(os.path.abspath(os.fspath(file)))
        except Exception:
            ap = ""
        if ap == _target:
            raise FileNotFoundError(2, "No such file or directory", str(file))
        return _real_open(file, *a, **kw)

    os.path.exists = _fake_exists
    builtins.open = _fake_open

# ---- ② 屏蔽模块：meta_path finder 直接拒绝 ----
if BLOCK_MODULE:
    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            root = fullname.split(".")[0]
            if fullname == BLOCK_MODULE or root == BLOCK_MODULE:
                raise ModuleNotFoundError(
                    "No module named %r (simulated missing)" % fullname,
                    name=fullname)
            return None
    sys.meta_path.insert(0, _Blocker())

os.chdir(r"E:\GPT-SoVITS")
for p in (r"E:\GPT-SoVITS", r"E:\GPT-SoVITS\GPT_SoVITS"):
    if p not in sys.path:
        sys.path.insert(0, p)

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
c = TTS_Config(); c.device = "cpu"; c.is_half = False
c.vits_weights_path = r"E:\GPT-SoVITS\GPT_SoVITS\pretrained_models\s2Gv3.pth"
c.t2s_weights_path  = r"E:\GPT-SoVITS\GPT_SoVITS\pretrained_models\s1v3.ckpt"
t = TTS(c)
sr, a = list(t.run({{
    "text": "\u8fd9\u662f\u4f60\u7684\u9a6c\u514b\u676f",
    "text_lang": "zh",
    "ref_audio_path": r"E:\cv\models\gpt_sovits\ref_audio.wav",
    "prompt_text": "\u5bf9\u6211\u505a\u4e86\u4ecb\u7ecd\u554a\u90a3\u4e48\u6211\u60f3\u8bf4\u7684\u8bdd\u662f\u5462\u5929\u5bb6\u5982\u679c\u5bf9\u6211\u7684\u7814\u7a76\u611f\u5174\u8da3\u5462\u4f60",
    "prompt_lang": "zh",
    "text_split_method": "cut0", "top_k": 15, "speed_factor": 1.0,
}}))[0]
assert len(a) > 1000, "audio too short: %d" % len(a)
print("PROBE_OK", len(a), sr)
'''


def run(block_module=None, block_file=None):
    code = TEMPLATE.format(block_module=block_module, block_file=block_file)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
               PYTHONPYCACHEPREFIX=os.path.join(tempfile.gettempdir(),
                                                "gsv_probe_cache"))
    try:
        p = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env, timeout=600)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    out = (p.stdout or "") + (p.stderr or "")
    if "PROBE_OK" in out:
        return True, [l for l in out.splitlines() if "PROBE_OK" in l][-1].strip()
    errs = [l.strip() for l in out.splitlines() if "Error" in l or "error" in l]
    return False, (errs[-1][:95] if errs else "rc=%d" % p.returncode)


def main():
    if not os.path.isfile(PY):
        print("[x] 找不到 %s" % PY)
        return 2

    print("基线检查（不屏蔽任何东西）...")
    ok, msg = run()
    if not ok:
        print("[x] 基线不通过，先修好再测: %s" % msg)
        return 2
    print("  基线通过: %s\n" % msg)

    print("逐项模拟「文件/模块不存在」...")
    rows = []
    for label, kind, target in CASES:
        bm = target if kind == "module" else None
        bf = target if kind == "file" else None
        if kind == "file" and not os.path.exists(bf):
            rows.append((label, "不存在", "-"))
            print("  %-34s 不存在" % label)
            continue
        ok, msg = run(bm, bf)
        # run() 返回 False = 模拟缺失后跑挂了 = 该项是必需的
        verdict = "必需" if not ok else "可删"
        rows.append((label, verdict, msg if not ok else ""))
        print("  %-34s %s" % (label, verdict))

    print("\n" + "=" * 86)
    print("%-34s %-8s %s" % ("目标", "结论", "模拟缺失后的报错"))
    print("=" * 86)
    for label, verdict, note in rows:
        print("%-34s %-8s %s" % (label, verdict, note))
    print("=" * 86)
    need = [r for r in rows if r[1] == "必需"]
    drop = [r for r in rows if r[1] == "可删"]
    print("必需 %d 项 / 本测试下可删 %d 项" % (len(need), len(drop)))
    if drop:
        print("注意：'可删' 只代表「当前这条短中文测试用不到」。")
        print("      长文本 / 含数字 / 英文场景可能仍需要，动前按场景复测。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
