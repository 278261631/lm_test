@echo off
setlocal
cd /d "%~dp0"

if not exist "lm_env\Scripts\python.exe" (
  echo [ERROR] lm_env not found.
  echo Please create venv first.
  exit /b 1
)

echo [RUN] Start GPU training
call "lm_env\Scripts\python.exe" train.py --device gpu --max-samples 0
echo [DONE] Training finished
