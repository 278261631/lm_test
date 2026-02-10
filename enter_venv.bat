@echo off
setlocal
cd /d "%~dp0"

if not exist "lm_env\Scripts\activate.bat" (
  echo [ERROR] 未找到虚拟环境: lm_env
  echo 请先执行: python -m venv lm_env
  pause
  exit /b 1
)

echo [OK] 正在进入虚拟环境...
call "lm_env\Scripts\activate.bat"
echo [OK] 已进入虚拟环境: %VIRTUAL_ENV%
echo.
echo 你可以手动安装依赖:
echo pip install -r requirements.txt
echo.
cmd /k
