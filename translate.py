import argparse
import math
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"


def tokenize_zh(text: str) -> List[str]:
    return [ch for ch in text.strip() if ch.strip()]


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


def encode_src(text: str, src_vocab: dict, max_src_len: int) -> List[int]:
    tokens = tokenize_zh(text)[: max_src_len - 2]
    ids = [src_vocab[BOS]] + [src_vocab.get(t, src_vocab[UNK]) for t in tokens] + [src_vocab[EOS]]
    if len(ids) < max_src_len:
        ids.extend([src_vocab[PAD]] * (max_src_len - len(ids)))
    return ids


def greedy_decode(
    model: nn.Module,
    src_ids: torch.Tensor,
    src_pad_id: int,
    tgt_bos_id: int,
    tgt_eos_id: int,
    tgt_pad_id: int,
    max_tgt_len: int,
    device: torch.device,
) -> List[int]:
    model.eval()
    generated = [tgt_bos_id]
    with torch.no_grad():
        for _ in range(max_tgt_len - 1):
            tgt_input = torch.tensor([generated], dtype=torch.long, device=device)
            src_key_padding_mask = src_ids.eq(src_pad_id)
            tgt_key_padding_mask = tgt_input.eq(tgt_pad_id)
            logits = model(src_ids, tgt_input, src_key_padding_mask, tgt_key_padding_mask)
            next_id = logits[:, -1, :].argmax(dim=-1).item()
            generated.append(next_id)
            if next_id == tgt_eos_id:
                break
    return generated


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

    parser = argparse.ArgumentParser(description="Run translation with trained Transformer model")
    parser.add_argument("--model-path", type=str, default="artifacts/best_model.pt")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--max-src-len", type=int, default=80)
    parser.add_argument("--max-tgt-len", type=int, default=80)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "gpu"])
    args = parser.parse_args()

    ckpt = torch.load(Path(args.model_path), map_location="cpu")
    src_vocab = ckpt["src_vocab"]
    tgt_vocab = ckpt["tgt_vocab"]
    cfg = ckpt.get("config", {})
    idx_to_tgt = {v: k for k, v in tgt_vocab.items()}

    model = Seq2SeqTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=cfg.get("d_model", 256),
        nhead=cfg.get("nhead", 8),
        num_encoder_layers=cfg.get("num_layers", 3),
        num_decoder_layers=cfg.get("num_layers", 3),
        dim_feedforward=cfg.get("ff_dim", 1024),
        dropout=cfg.get("dropout", 0.1),
        max_len=max(args.max_src_len, args.max_tgt_len),
    )
    model.load_state_dict(ckpt["model_state_dict"])

    if args.device in ("cuda", "gpu"):
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    src_ids = torch.tensor(
        [encode_src(args.text, src_vocab, args.max_src_len)],
        dtype=torch.long,
        device=device,
    )
    pred_ids = greedy_decode(
        model=model,
        src_ids=src_ids,
        src_pad_id=src_vocab[PAD],
        tgt_bos_id=tgt_vocab[BOS],
        tgt_eos_id=tgt_vocab[EOS],
        tgt_pad_id=tgt_vocab[PAD],
        max_tgt_len=args.max_tgt_len,
        device=device,
    )
    tokens = []
    for tid in pred_ids:
        tok = idx_to_tgt.get(tid, "")
        if tok in (BOS, EOS, PAD):
            continue
        tokens.append(tok)
    print("".join(tokens))


if __name__ == "__main__":
    main()
