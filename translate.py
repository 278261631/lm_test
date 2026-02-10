import argparse
import json
import sys
from pathlib import Path
from typing import Dict


def translate(src_text: str, exact_map: Dict[str, str], char_map: Dict[str, str], default_tgt_char: str) -> str:
    if src_text in exact_map:
        return exact_map[src_text]
    out = []
    for ch in src_text:
        if not ch.strip():
            continue
        out.append(char_map.get(ch, default_tgt_char))
    return "".join(out)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

    parser = argparse.ArgumentParser(description="使用统计翻译基线模型进行中文->维语翻译")
    parser.add_argument("--model-path", type=str, default="artifacts/char_mapping_model.json")
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    with model_path.open("r", encoding="utf-8") as f:
        model = json.load(f)

    pred = translate(
        src_text=args.text,
        exact_map=model.get("exact_map", {}),
        char_map=model.get("char_map", {}),
        default_tgt_char=model.get("default_tgt_char", ""),
    )
    print(pred)


if __name__ == "__main__":
    main()
