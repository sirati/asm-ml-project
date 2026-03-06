from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

from data.text_dataset import create_minibatch_loaders
from model.architectures.mamba_flash_hybrid import (
    MambaFlashHybridConfig,
    MambaFlashHybridModel,
)
from training.config import MinibatchEntry, TrainConfig
from training.loop import train_loop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-subset-splits", type=int, default=4)
    parser.add_argument(
        "--minibatch",
        type=str,
        nargs="+",
        default=["512:4"],
        help="seq_len:batch_size pairs, e.g. 256:8 512:4 1024:2",
    )
    parser.add_argument("--estimate-memory", action="store_true")
    args = parser.parse_args()

    entries = []
    for spec in args.minibatch:
        seq_len_str, batch_str = spec.split(":")
        entries.append(
            MinibatchEntry(seq_len=int(seq_len_str), batch_size=int(batch_str))
        )

    max_seq_len = max(e.seq_len for e in entries)

    train_config = TrainConfig(
        max_steps=args.max_steps,
        learning_rate=args.lr,
        device=args.device,
        minibatch_spec=entries,
    )

    tokenizer = AutoTokenizer.from_pretrained(train_config.tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config = MambaFlashHybridConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=max_seq_len,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_subset_splits=args.num_subset_splits,
    )

    device = torch.device(train_config.device)

    if args.estimate_memory:
        # Build model on CPU for estimation (no CUDA needed)
        model = MambaFlashHybridModel(model_config)
        for entry in entries:
            est = model.estimate_memory(
                batch_size=entry.batch_size,
                seq_len=entry.seq_len,
                training=True,
            )
            print(f"--- seq_len={entry.seq_len}, batch_size={entry.batch_size} ---")
            print(est.summary())
            print()
        return

    model = MambaFlashHybridModel(model_config).to(device)

    loaders = create_minibatch_loaders(
        entries=entries,
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_token_id,
    )

    train_loop(model, loaders, train_config)


if __name__ == "__main__":
    main()
