@echo off
setlocal
cd /d "%~dp0"

if not exist "lm_env\Scripts\python.exe" (
  echo [ERROR] 未找到虚拟环境: lm_env
  pause
  exit /b 1
)

if not exist "artifacts\char_mapping_model.json" (
  echo [ERROR] 未找到模型文件: artifacts\char_mapping_model.json
  echo 请先运行训练脚本。
  pause
  exit /b 1
)

set "TEXT=创建一个新的记录"
if not "%~1"=="" set "TEXT=%~1"

echo [RUN] 测试翻译文本: %TEXT%
call "lm_env\Scripts\python.exe" translate.py --model-path artifacts/char_mapping_model.json --text "%TEXT%"
echo.
echo [DONE] 测试结束
pause
