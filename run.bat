@echo off
rem Launch with vision_ai env (base python lacks openvino/onnxruntime)
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set PY=C:\Users\admin\anaconda3\envs\vision_ai\python.exe
if not exist "%PY%" (
  echo vision_ai python not found: %PY%
  exit /b 1
)
"%PY%" "%~dp0main.py" %*
exit /b %ERRORLEVEL%
