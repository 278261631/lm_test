import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_parallel_data(tsv_path: Path, max_samples: int = 0) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with tsv_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            src, tgt = parts[0].strip(), parts[1].strip()
            if not src or not tgt:
                continue
            pairs.append((src, tgt))
            if max_samples > 0 and len(pairs) >= max_samples:
                break
    return pairs


def build_char_mapping(train_pairs: List[Tuple[str, str]]) -> Tuple[Dict[str, Dict[str, int]], str]:
    mapping_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tgt_char_counter: Counter = Counter()

    for src_text, tgt_text in train_pairs:
        src_chars = [ch for ch in src_text if ch.strip()]
        tgt_chars = [ch for ch in tgt_text if ch.strip()]
        if not src_chars or not tgt_chars:
            continue

        tgt_char_counter.update(tgt_chars)

        if len(src_chars) == 1:
            mapping_counts[src_chars[0]][tgt_chars[0]] += 1
            continue

        src_last = len(src_chars) - 1
        tgt_last = len(tgt_chars) - 1
        for i, s_ch in enumerate(src_chars):
            j = round(i * tgt_last / src_last) if src_last > 0 else 0
            t_ch = tgt_chars[min(max(j, 0), tgt_last)]
            mapping_counts[s_ch][t_ch] += 1

    default_tgt_char = tgt_char_counter.most_common(1)[0][0] if tgt_char_counter else ""
    return mapping_counts, default_tgt_char


def best_mapping(mapping_counts: Dict[str, Dict[str, int]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for s_ch, tgt_count in mapping_counts.items():
        best_t = max(tgt_count.items(), key=lambda x: x[1])[0]
        mapping[s_ch] = best_t
    return mapping


def translate_with_mapping(
    src_text: str,
    exact_map: Dict[str, str],
    char_map: Dict[str, str],
    default_tgt_char: str,
) -> str:
    if src_text in exact_map:
        return exact_map[src_text]
    out = []
    for ch in src_text:
        if not ch.strip():
            continue
        out.append(char_map.get(ch, default_tgt_char))
    return "".join(out)


def char_overlap_score(pred: str, gold: str) -> float:
    pred_chars = [c for c in pred if c.strip()]
    gold_chars = [c for c in gold if c.strip()]
    if not pred_chars or not gold_chars:
        return 0.0
    pred_counter = Counter(pred_chars)
    gold_counter = Counter(gold_chars)
    hit = sum(min(pred_counter[k], gold_counter[k]) for k in pred_counter)
    p = hit / max(len(pred_chars), 1)
    r = hit / max(len(gold_chars), 1)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

    parser = argparse.ArgumentParser(description="中文->维语 统计翻译基线模型训练（纯 Python）")
    parser.add_argument("--data-path", type=str, default="data/zh_ug_parallel.tsv")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--max-samples", type=int, default=0, help="0 表示全量数据")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu", "cuda"])
    parser.add_argument("--disable-exact-map", action="store_true", help="禁用整句记忆映射")
    args = parser.parse_args()

    if args.device in ("gpu", "cuda"):
        print("提示: 当前脚本为统计基线实现，不使用 GPU，实际仍为 CPU 训练。")

    random.seed(args.seed)
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_parallel_data(data_path, args.max_samples)
    if len(pairs) < 100:
        raise ValueError(f"样本太少: {len(pairs)}，请检查数据。")

    random.shuffle(pairs)
    split_idx = int(len(pairs) * (1 - args.val_ratio))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    mapping_counts, default_tgt_char = build_char_mapping(train_pairs)
    char_map = best_mapping(mapping_counts)
    exact_map = {} if args.disable_exact_map else {s: t for s, t in train_pairs}

    scores: List[float] = []
    for src, tgt in val_pairs:
        pred = translate_with_mapping(src, exact_map, char_map, default_tgt_char)
        scores.append(char_overlap_score(pred, tgt))
    avg_score = sum(scores) / max(len(scores), 1)

    model = {
        "type": "char_mapping_baseline",
        "data_path": str(data_path),
        "train_size": len(train_pairs),
        "val_size": len(val_pairs),
        "val_char_f1_like": round(avg_score, 6),
        "default_tgt_char": default_tgt_char,
        "char_map": char_map,
        "exact_map": exact_map,
    }

    model_path = output_dir / "char_mapping_model.json"
    with model_path.open("w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False)

    print(f"训练完成，模型已保存: {model_path}")
    print(f"训练集: {len(train_pairs)}  验证集: {len(val_pairs)}")
    print(f"验证集字符级 F1-like: {avg_score:.4f}")


if __name__ == "__main__":
    main()
