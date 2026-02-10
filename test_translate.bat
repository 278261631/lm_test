@echo off
setlocal
cd /d "%~dp0"

if not exist "lm_env\Scripts\python.exe" (
  echo [ERROR] lm_env not found.
  exit /b 1
)

if not exist "artifacts\best_model.pt" (
  echo [ERROR] model not found: artifacts\best_model.pt
  echo Please run training first.
  exit /b 1
)

set "TEXT=test"
if not "%~1"=="" set "TEXT=%*"

echo [RUN] test text: %TEXT%
call "lm_env\Scripts\python.exe" translate.py --model-path artifacts/best_model.pt --text "%TEXT%" --device auto
echo [DONE] test finished
