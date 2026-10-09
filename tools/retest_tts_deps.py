"""复测：短句测不出来的依赖，用长文本 / 含数字 / 中英混排再验一次。

背景：非破坏性探测（probe_tts_deps_nondestructive.py）显示 s2.json、
zh_normalization、fast_langdetect 在「短中文」场景下用不到，
但那可能只是短句恰好没触发相应代码路径。本脚本换更丰富的输入复测。
"""
import os
import subprocess
import sys
import tempfile

PY = r"C:\Users\admin\anaconda3\envs\GPTSoVits\python.exe"

TPL = r'''
import os, sys, builtins
BF = "__BLOCK_FILE__"
if BF:
    _re = os.path.exists
    _ro = builtins.open
    _t = os.path.normcase(os.path.abspath(BF))

    def _fe(p):
        try:
            ap = os.path.normcase(os.path.abspath(os.fspath(p)))
        except Exception:
            return _re(p)
        return False if ap == _t else _re(p)

    def _fo(f, *a, **k):
        try:
            ap = os.path.normcase(os.path.abspath(os.fspath(f)))
        except Exception:
            ap = ""
        if ap == _t:
            raise FileNotFoundError(2, "No such file", str(f))
        return _ro(f, *a, **k)

    os.path.exists = _fe
    builtins.open = _fo

os.chdir(r"E:\GPT-SoVITS")
for p in (r"E:\GPT-SoVITS", r"E:\GPT-SoVITS\GPT_SoVITS"):
    if p not in sys.path:
        sys.path.insert(0, p)

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
c = TTS_Config(); c.device = "cpu"; c.is_half = False
c.vits_weights_path = r"E:\GPT-SoVITS\GPT_SoVITS\pretrained_models\s2Gv3.pth"
c.t2s_weights_path = r"E:\GPT-SoVITS\GPT_SoVITS\pretrained_models\s1v3.ckpt"
t = TTS(c)

LONG = ("\u8fd9\u662f\u4f60\u7684\u9a6c\u514b\u676f\uff0c\u76f8\u4f3c\u5ea6\u767e\u5206\u4e4b\u516b\u5341\u4e09\uff0c"
        "\u5df2\u7ecf\u8bc6\u522b\u51fa\u6765\u4e86\uff0c\u5e76\u4e14\u6211\u4eec\u8fd8\u652f\u6301\u8bed\u97f3\u4ea4\u4e92"
        "\u529f\u80fd\uff0c\u5305\u62ec\u8bc6\u522b\u3001\u8bb0\u5f55\u548c\u641c\u7d22\u4e09\u79cd\u6a21\u5f0f\uff0c"
        "\u6570\u5b57\u8bf4\u660e\u5982\u4e0b\uff1a\u5341\u4e8c\u3001\u4e8c\u5341\u3001\u4e09\u5341\u56db\u3002")
MIXED = ("\u8fd9\u662f\u4f60\u7684 ResNet18 \u7279\u5f81\u68c0\u7d22\uff0c\u76f8\u4f3c\u5ea6 0.83\uff0c"
         "GPU \u63a8\u7406\u3002")
SHORT = "\u8fd9\u662f\u4f60\u7684\u9a6c\u514b\u676f\u3002"
PROMPT = ("\u5bf9\u6211\u505a\u4e86\u4ecb\u7ecd\u554a\u90a3\u4e48\u6211\u60f3\u8bf4\u7684\u8bdd\u662f\u5462"
          "\u5929\u5bb6\u5982\u679c\u5bf9\u6211\u7684\u7814\u7a76\u611f\u5174\u8da3\u5462\u4f60")

for name, txt, method in (("short", SHORT, "cut0"),
                          ("long_num", LONG, "cut1"),
                          ("mixed_en", MIXED, "cut1")):
    try:
        sr, a = list(t.run({"text": txt, "text_lang": "zh",
                            "ref_audio_path": r"E:\cv\models\gpt_sovits\ref_audio.wav",
                            "prompt_text": PROMPT,
                            "prompt_lang": "zh", "text_split_method": method,
                            "top_k": 15, "speed_factor": 1.0}))[0]
        print("OK", name, len(a), sr)
    except Exception as e:
        print("FAIL", name, type(e).__name__, str(e)[:90])
'''

CASES = [
    ("baseline", None),
    ("s2.json", r"E:\GPT-SoVITS\GPT_SoVITS\configs\s2.json"),
    ("zh_normalization", r"E:\GPT-SoVITS\GPT_SoVITS\text\zh_normalization\text_normlization.py"),
    ("fast_langdetect", r"E:\GPT-SoVITS\GPT_SoVITS\pretrained_models\fast_langdetect\lid.176.bin"),
]


def main():
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
               PYTHONPYCACHEPREFIX=os.path.join(tempfile.gettempdir(), "gsv_rt"))
    for label, block_file in CASES:
        code = TPL.replace("__BLOCK_FILE__", repr(block_file) if block_file else '""')
        try:
            p = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", env=env, timeout=900)
        except subprocess.TimeoutExpired:
            print("### %-20s TIMEOUT" % label)
            continue
        out = (p.stdout or "") + (p.stderr or "")
        lines = [l for l in out.splitlines() if l.startswith(("OK", "FAIL"))]
        print("### %s" % label)
        for l in lines:
            print("   ", l[:100])
        if not lines:
            tail = (out.strip().splitlines() or ["<empty>"])[-1]
            print("    (no output)", tail[:100])
    return 0


if __name__ == "__main__":
    sys.exit(main())
