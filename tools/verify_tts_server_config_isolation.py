#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证 tts_server --vits-weights 不会污染 configs/tts_infer.yaml。

背景
----
TTS_Config.__init__ 会调save_configs() 就地重写 GPT_SoVITS/configs/tts_infer.yaml。
所以 `tts_server --vits-weights <临时路径>` 启动一次之后，
下一次裸跑 start_tts_server.bat 就会去加载那个已经被删掉的临时文件 → FileNotFoundError。
（实测踩过：yaml 里真的残留了 Temp/e2e_export_xxx/my_voice_s2Gv3.pth）

本脚本验证 voice/tts_server.py 里的还原逻辑：
  1. 备份 yaml
  2. 用一个「不同于默认路径」的权重起服务
  3. 真的发一次 POST 合成（确认引擎可用，不只是构造成功）
  4. 关服务，比对 yaml 是否与备份一致
"""
from __future__ import annotations

import filecmp
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAML = r"E:\GPT-SoVITS\GPT_SoVITS\configs\tts_infer.yaml"
VITS_ROOT = r"E:\GPT-SoVITS"
PORT = 19897
REF_AUDIO = os.path.join(ROOT, "models", "gpt_sovits", "ref_audio.wav")

# 代理会劫持 localhost，必须显式关掉（见项目记忆：TtsClient 的 ProxyHandler({})）
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def main() -> int:
    # 造一个「不在默认位置」的权重副本，确保能测出 yaml 是否被改写
    fake = os.path.join(tempfile.gettempdir(), "srv_probe_vits.pth")
    shutil.copy2(os.path.join(VITS_ROOT, "GPT_SoVITS", "pretrained_models",
                              "s2Gv3.pth"), fake)
    print("测试权重(副本): %s" % fake)

    bak = fake + ".yamlbak"
    shutil.copy2(YAML, bak)
    before = open(YAML, encoding="utf-8").read()
    print("启动前 yaml vits 行: %s"
          % [ln for ln in before.splitlines() if "vits_weights_path" in ln][0])

    proc = subprocess.Popen(
        [sys.executable, "-m", "voice.tts_server",
         "--gpt-sovits-path", VITS_ROOT, "--port", str(PORT),
         "--vits-weights", fake, "--ref-audio", REF_AUDIO],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})

    ok = False
    try:
        # 等引擎就绪（预热约 6s）
        for _ in range(90):
            if proc.poll() is not None:
                print("[x] 服务提前退出")
                print(proc.stdout.read()[-1500:])
                return 1
            try:
                with _OPENER.open("http://127.0.0.1:%d/health" % PORT,
                                  timeout=2) as r:
                    if json.loads(r.read().decode()).get("ok"):
                        print("引擎就绪")
                        ok = True
                        break
            except Exception:
                time.sleep(1)
        if not ok:
            print("[x] 等待引擎就绪超时")
            return 1

        # 真的合成一次
        # v3 走SoVITS 分支，prompt_text / prompt_lang 必填（引擎会直接抛
        # "prompt_text cannot be empty when using SoVITS_V3"）
        body = json.dumps({
            "text": "服务端权重覆盖测试。",
            "prompt_text": "欢迎使用语音交互物体识别系统。",
            "prompt_lang": "zh",
            "text_lang": "zh",
        }).encode()
        req = urllib.request.Request("http://127.0.0.1:%d/tts" % PORT,
                                     data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with _OPENER.open(req, timeout=180) as r:
            # 默认 return_path=False，直接返回 audio/wav 字节流而非 JSON
            data = r.read()
            ctype = r.headers.get("Content-Type", "")
        # WAV 头：RIFF....WAVE
        ok_wav = data[:4] == b"RIFF" and data[8:12] == b"WAVE"
        print("合成成功: %.1fs耗时, %d 字节, Content-Type=%s, RIFF/WAVE 头正确=%s"
              % (time.time() - t0, len(data), ctype, ok_wav))
        if len(data) < 10000 or not ok_wav:
            print("[x] 音频内容异常")
            return 1
    finally:
        proc.terminate()
        try:
            out = proc.stdout.read()
        except Exception:
            out = ""
        proc.wait(timeout=20)

    for ln in (out or "").splitlines():
        if "engine ready" in ln or "config restored" in ln or "WARN" in ln:
            print("  " + ln.strip())

    print("\n服务退出后比对 yaml:")
    after = open(YAML, encoding="utf-8").read()
    print("  当前 vits 行: %s"
          % [ln for ln in after.splitlines() if "vits_weights_path" in ln][0])
    if filecmp.cmp(YAML, bak, shallow=False):
        print("  [OK] yaml 与启动前完全一致 —— 还原逻辑生效")
        rc = 0
    else:
        print("  [x] yaml 被改写，未还原！")
        import difflib
        for ln in difflib.unified_diff(before.splitlines(), after.splitlines(),
                                        "before", "after", lineterm="", n=0):
            print("    " + ln)
        rc = 1

    os.remove(fake)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())