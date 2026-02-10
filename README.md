# 中文-维语翻译模型训练

本项目使用 `data/zh_ug_parallel.tsv`（中文\t维语）训练一个**纯 Python 统计翻译基线模型**（离线可跑、无需第三方库）。

## 1. 创建虚拟环境（Windows PowerShell）

```powershell
python -m venv lm_env
.\lm_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. 开始训练

先用小样本验证流程：

```powershell
python train.py --max-samples 5000
```

正式训练（全量数据）：

```powershell
python train.py --max-samples 0
```

训练输出：
- 模型文件：`artifacts/char_mapping_model.json`
- 终端会输出验证集字符级 F1-like 指标（粗粒度）

## 3. 推理测试

```powershell
python translate.py --model-path artifacts/char_mapping_model.json --text "创建一个新的记录"
```

## 可调参数

`train.py` 常用参数：
- `--max-samples`：使用的数据条数，`0` 表示全量。
- `--val-ratio`：验证集占比（默认 0.05）。
- `--disable-exact-map`：禁用整句记忆映射（更像“纯泛化”测试）。
