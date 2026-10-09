#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""端到端验证「训练权重 -> 导出纯推理权重 -> 引擎加载 -> 实际合成」。

回答的问题
----------
「语料训练出来的权重，能不能不依赖训练状态、直接丢给 tts_server 合成？」

本脚本做三件事，每步都打印可核对的数字：
  1. 造一个「训练断点形态」的文件：{model, iteration, optimizer, learning_rate}
     —— 与 utils.save_checkpoint() 的真实输出一致（optimizer 是主要体积来源），
     并刻意不含 config（真实 save_checkpoint 也不存）
  2. 用 tools/export_infer_weights.py 剥离成 {weight, config, info}
  3. 用导出后的文件启动真实 GPT-SoVITS 引擎合成一句中文，回读音频指标

注意：这里用官方底模s2Gv3.pth 当「训练产物」来模拟，不需要真的训练一次。
     两者在文件结构上的差别只有 optimizer 那一份 state_dict，
     而导出脚本只取 model/weight + config，对两者一视同仁。

用法
----
    C:\\Users\\admin\\anaconda3\\envs\\GPTSoVits\\python.exe tools\\verify_export_pipeline.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VITS_ROOT = r"E:\GPT-SoVITS"
BASE_MODEL = os.path.join(VITS_ROOT, "GPT_SoVITS", "pretrained_models", "s2Gv3.pth")
PROBE = os.path.join(ROOT, "tools", "export_infer_weights.py")
# 参考音：合成必须给一段参考音频，GPT-SoVITS 靠它克隆音色
REF_AUDIO = os.path.join(ROOT, "models", "gpt_sovits", "ref_audio.wav")

# 被测脚本。用 .replace 注入路径而非 % 格式化 —— 里面有 tqdm 进度条和
# print 的 % 格式符，走 % 会被迫写成 %% 再被外层吃掉，已连续踩两次。
_PROBE_TMPL = r'''
import os, sys, time, shutil, tempfile

root = r"__ROOT__"
os.chdir(root)
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "GPT_SoVITS"))

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

# TTS_Config.__init__ 会 save_configs() 就地重写 configs/tts_infer.yaml，
# 会把本次的 version / 权重路径固化进共享配置。先备份，测完还原。
y = os.path.join(root, "GPT_SoVITS", "configs", "tts_infer.yaml")
bak = os.path.join(tempfile.gettempdir(), "tts_infer.yaml.bak")
shutil.copy2(y, bak)

cfg = TTS_Config()
print("YAML_VERSION=%s" % cfg.version)
cfg.device = "cpu"
cfg.is_half = False
cfg.version = "v3"
cfg.update_version("v3")
cfg.vits_weights_path = r"__WEIGHTS__"
cfg.t2s_weights_path = os.path.join(root, "GPT_SoVITS", "pretrained_models", "s1v3.ckpt")

t0 = time.time()
tts = TTS(cfg)
t1 = time.time()
shutil.copy2(bak, y)

inputs = {
    "text": "导出纯推理权重后直接合成。",
    "text_lang": "zh",
    "ref_audio_path": r"__REFAUDIO__",
    "prompt_text": "欢迎使用语音交互物体识别系统。",
    "prompt_lang": "zh",
    "text_split_method": "cut5",
    "top_k": 15, "top_p": 1.0, "temperature": 1.0,
    "speed_factor": 1.0, "fragment_interval": 0.3, "seed": -1,
}
st = time.time()
res = tts.run(inputs)
if isinstance(res, tuple) and len(res) == 2:
    sr, audio = res
else:
    chunks = list(res)
    sr, audio = chunks[0]

samples = list(audio)
peak = max(abs(int(v)) for v in samples) if len(samples) else 0
print("SR=%d DUR=%.2f PEAK=%d INIT=%.1f SYNTH=%.1f VITS=%s"
      % (sr, len(samples) / float(sr), peak, t1 - t0, time.time() - st,
         os.path.basename(cfg.vits_weights_path)))
shutil.copy2(bak, y)
'''


def mb(n: int) -> str:
    return "%.1f MB" % (n / 1024 / 1024)


def step1_make_fake_training_ckpt(dst: str) -> None:
    import torch

    print("[1/3] 构造训练断点形态文件（模拟 utils.save_checkpoint 输出）")
    print("      源码依据：GPT_SoVITS/utils.py save_checkpoint() -> "
          "{model, iteration, optimizer, learning_rate}")
    base = torch.load(BASE_MODEL, map_location="cpu", weights_only=False)
    weight = base.get("weight", base.get("model"))
    weight = {k: v.float() for k, v in weight.items()}   # 训练产物是 fp32

    # 训练断点里还有判别器的 enc_q 分支，官方 savee() 会剔除，这里保留以验证
    weight["enc_q.weight"] = torch.zeros(4, 4)

    # 造一份有真实体积的optimizer 状态：Adam 的 exp_avg / exp_avg_sq 各一份
    # 只取前 40 个张量，够撑出体积又不会白烧几分钟
    opt_state = {}
    for i, (k, v) in enumerate(weight.items()):
        if i >= 40:
            break
        opt_state[k] = {"step": 0, "exp_avg": torch.zeros_like(v),
                        "exp_avg_sq": torch.zeros_like(v)}

    torch.save({"model": weight,
                "iteration": 12000,
                "optimizer": {"state": opt_state, "param_groups": []},
                "learning_rate": 1e-4,
                "scaler_state_dict": {}}, dst)
    print("      训练断点: %s (%s)" % (os.path.basename(dst), mb(os.path.getsize(dst))))
    print("      顶层键  : model / iteration / optimizer / learning_rate")
    print("      刻意不含 config —— 真实 save_checkpoint() 也不存")


def step2_export(src: str, dst: str) -> None:
    print("\n[2/3] 调用 export_infer_weights.py 剥离")
    before = os.path.getsize(src)
    r = subprocess.run([sys.executable, PROBE, "--src", src, "--dst", dst],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    for line in (r.stdout or "").splitlines():
        print("      " + line)
    if r.returncode != 0:
        print("      导出失败: %s" % ((r.stderr or "")[-500:],))
        sys.exit(1)
    after = os.path.getsize(dst)
    print("      瘦身 %.1f%%  (%s -> %s)"
          % ((1 - after / before) * 100, mb(before), mb(after)))


def step3_synthesize(weights: str, tag: str) -> None:
    print("\n[3/3] 用该权重启动真实引擎合成（tag=%s）" % tag)
    print("      加载方式对齐 voice/tts_server.py:115-152：")
    print("        chdir(<root>) + sys.path 同时加 <root> 和 <root>/GPT_SoVITS")
    print("        config.vits_weights_path = 导出文件")
    print("      GPT_SoVITS/TTS_infer_pack/TTS.py:505-506")
    print("        dict_s2 = load_sovits_new(weights_path)")
    print("        hps = dict_s2[\"config\"]  <- 无条件读取，缺 config 直接 KeyError")

    probe = (_PROBE_TMPL.replace("__ROOT__", VITS_ROOT)
             .replace("__WEIGHTS__", weights)
             .replace("__REFAUDIO__", REF_AUDIO))

    p = os.path.join(tempfile.gettempdir(), "e2e_tts_probe.py")
    with open(p, "w", encoding="utf-8") as f:
        f.write(probe)

    r = subprocess.run([sys.executable, p], capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8",
                            "PYTHONUTF8": "1"})
    out = (r.stdout or "")
    for line in out.splitlines():
        if line.startswith(("SR=", "YAML_VERSION=")):
            print("      " + line)
    if "SR=" not in out:
        tail = [ln for ln in (r.stderr or "").splitlines() if ln.strip()][-6:]
        print("      合成失败，stderr 末尾:")
        for line in tail:
            print("        " + line)
        sys.exit(1)


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="e2e_export_")
    fake_ckpt = os.path.join(tmp, "G_12000.pth")
    infer_ckpt = os.path.join(tmp, "my_voice_s2Gv3.pth")

    step1_make_fake_training_ckpt(fake_ckpt)
    step2_export(fake_ckpt, infer_ckpt)
    step3_synthesize(infer_ckpt, "exported")

    print("\n结论：训练断点剥离后可直接被 TTS.py 加载并合成，无需再次训练。")
    print("产物: %s" % infer_ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())