import argparse
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize_zh(text: str) -> List[str]:
    return [ch for ch in text.strip() if ch.strip()]


def tokenize_ug(text: str) -> List[str]:
    return [ch for ch in text.strip() if ch.strip()]


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


def build_vocab(tokenized_texts: List[List[str]], min_freq: int = 1) -> dict:
    counter: Counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)
    vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
    for tok, freq in counter.items():
        if freq >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def encode(tokens: List[str], vocab: dict, max_len: int) -> List[int]:
    ids = [vocab[BOS]]
    ids.extend(vocab.get(t, vocab[UNK]) for t in tokens[: max_len - 2])
    ids.append(vocab[EOS])
    if len(ids) < max_len:
        ids.extend([vocab[PAD]] * (max_len - len(ids)))
    return ids


class ParallelDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        src_vocab: dict,
        tgt_vocab: dict,
        max_src_len: int,
        max_tgt_len: int,
    ) -> None:
        self.items = []
        for src_text, tgt_text in pairs:
            src_ids = encode(tokenize_zh(src_text), src_vocab, max_src_len)
            tgt_ids = encode(tokenize_ug(tgt_text), tgt_vocab, max_tgt_len)
            self.items.append((torch.tensor(src_ids), torch.tensor(tgt_ids)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt_input: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        src_emb = self.pos_enc(self.src_emb(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.tgt_emb(tgt_input) * math.sqrt(self.d_model))
        tgt_len = tgt_input.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt_input.device)
        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.fc_out(out)


@dataclass
class TrainConfig:
    data_path: str
    output_dir: str
    max_samples: int
    max_src_len: int
    max_tgt_len: int
    min_freq: int
    val_ratio: float
    batch_size: int
    epochs: int
    lr: float
    d_model: int
    nhead: int
    num_layers: int
    ff_dim: int
    dropout: float
    seed: int
    device: str


def resolve_device(device_arg: str) -> torch.device:
    arg = device_arg.lower().strip()
    if arg in ("gpu", "cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but CUDA is not available. Check torch CUDA installation.")
        return torch.device("cuda")
    if arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    data_path = Path(cfg.data_path)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_parallel_data(data_path, cfg.max_samples)
    if len(pairs) < 100:
        raise ValueError(f"Too few samples: {len(pairs)}")

    random.shuffle(pairs)
    split_idx = int(len(pairs) * (1 - cfg.val_ratio))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    src_tokens = [tokenize_zh(s) for s, _ in train_pairs]
    tgt_tokens = [tokenize_ug(t) for _, t in train_pairs]
    src_vocab = build_vocab(src_tokens, min_freq=cfg.min_freq)
    tgt_vocab = build_vocab(tgt_tokens, min_freq=cfg.min_freq)
    src_pad_id = src_vocab[PAD]
    tgt_pad_id = tgt_vocab[PAD]

    train_ds = ParallelDataset(train_pairs, src_vocab, tgt_vocab, cfg.max_src_len, cfg.max_tgt_len)
    val_ds = ParallelDataset(val_pairs, src_vocab, tgt_vocab, cfg.max_src_len, cfg.max_tgt_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = Seq2SeqTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_layers,
        num_decoder_layers=cfg.num_layers,
        dim_feedforward=cfg.ff_dim,
        dropout=cfg.dropout,
        max_len=max(cfg.max_src_len, cfg.max_tgt_len),
    ).to(device)

    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_val_loss = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [train]")
        for src, tgt in pbar:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            src_key_padding_mask = src.eq(src_pad_id)
            tgt_key_padding_mask = tgt_input.eq(tgt_pad_id)
            logits = model(src, tgt_input, src_key_padding_mask, tgt_key_padding_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(device)
                tgt = tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                src_key_padding_mask = src.eq(src_pad_id)
                tgt_key_padding_mask = tgt_input.eq(tgt_pad_id)
                logits = model(src, tgt_input, src_key_padding_mask, tgt_key_padding_mask)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / max(len(val_loader), 1)

        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "src_vocab": src_vocab,
                    "tgt_vocab": tgt_vocab,
                    "config": cfg.__dict__,
                },
                output_dir / "best_model.pt",
            )
            print(f"Saved best model: {output_dir / 'best_model.pt'}")

    print(f"Training done. Best val loss: {best_val_loss:.4f}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Chinese -> Uyghur Transformer training")
    parser.add_argument("--data-path", type=str, default="data/zh_ug_parallel.tsv")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--max-samples", type=int, default=30000, help="0 means all samples")
    parser.add_argument("--max-src-len", type=int, default=80)
    parser.add_argument("--max-tgt-len", type=int, default=80)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "gpu", "cuda"])
    args = parser.parse_args()
    return TrainConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        min_freq=args.min_freq,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
    config = parse_args()
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(config.output_dir) / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, ensure_ascii=False, indent=2)
    run_training(config)
