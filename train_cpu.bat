@echo off
setlocal
cd /d "%~dp0"

if not exist "lm_env\Scripts\python.exe" (
  echo [ERROR] lm_env not found.
  echo Please create venv first.
  exit /b 1
)

call "lm_env\Scripts\python.exe" -c "import torch" 1>nul 2>nul
if %ERRORLEVEL%==1 (
  echo [ERROR] torch is not installed in lm_env.
  echo Install: pip install -r requirements.txt
  exit /b 1
)

echo [RUN] Start CPU training
call "lm_env\Scripts\python.exe" train.py --device cpu --max-samples 0
echo [DONE] Training finished
