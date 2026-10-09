@echo off
rem ============================================================
rem  GPT-SoVITS TTS server launcher (选题一 语音播报)
rem
rem  必须用 GPTSoVits 环境运行 —— 该环境才有 torch/torchaudio/
rem  torchcodec/transformers。vision_ai 里没有 torchaudio。
rem
rem  启动后在另一个终端运行:  python main.py --camera 0
rem  健康检查:curl http://127.0.0.1:19880/health
rem ============================================================
setlocal
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

set GPT=C:\Users\admin\anaconda3\envs\GPTSoVits\python.exe
set GSVPATH=E:\GPT-SoVITS
set REF=%~dp0..\models\gpt_sovits\ref_audio.wav
set PROMPT=对我做了介绍啊那么我想说的话是呢大家如果对我的研究感兴趣呢你
set PORT=19880

if not exist "%GPT%" (
  echo [err] GPTSoVits python not found: %GPT%
  exit /b 1
)
if not exist "%REF%" (
  echo [err] reference audio not found: %REF%
  echo        需要 5-10s 清晰人声 wav（prompt_text 必须与音频内容一致）
  exit /b 1
)

rem 后台常驻：start /b 让服务器独立于本窗口
start "GPT-SoVITS TTS" /min "%GPT%" -m voice.tts_server ^
  --gpt-sovits-path "%GSVPATH%" ^
  --port %PORT% ^
  --device cpu ^
  --ref-audio "%REF%" ^
  --prompt-text "%PROMPT%" ^
  --vits-weights "%GSVPATH%\GPT_SoVITS\pretrained_models\s2Gv3.pth" ^
  --t2s-weights  "%GSVPATH%\GPT_SoVITS\pretrained_models\s1v3.ckpt" ^
  --preload

echo [ok] GPT-SoVITS TTS server starting on http://127.0.0.1:%PORT%
echo      首次 preload 约需 20-30 秒（加载 s1v3 + s2Gv3 + 中文HuBERT）
echo      另开终端运行: python main.py --camera 0
endlocal
