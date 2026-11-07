@echo off
rem Launch the OpenCV desktop UI.
rem Resolution order: %VISION_PYTHON% override -> .venv\Scripts\python.exe -> python on PATH.
rem Usage: run.bat --camera 0 --no-voice
setlocal
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set "PY=%VISION_PYTHON%"
if "%PY%"=="" if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if "%PY%"=="" set "PY=python"
"%PY%" "%~dp0main.py" %*
exit /b %ERRORLEVEL%
