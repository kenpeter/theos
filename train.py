"""
Training script for TinyModel — downloads code + chat + wiki data, BPE tokenizer.
Supports resume, checkpoint every N steps, keeps only latest + best.
"""

import os
import math
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tiny_model import TinyConfig, TinyModel
from bpe_tokenizer import BPETokenizer


def load_wikitext(split="train"):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split=split)
    return [r["text"] for r in ds if len(r["text"].strip()) >= 50]


def load_code_alpaca(split="train", max_samples=5000):
    from datasets import load_dataset
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split=split, streaming=False)
    texts = []
    for i, r in enumerate(ds):
        if i >= max_samples:
            break
        instr = r.get("instruction", "")
        inp = r.get("input", "")
        out = r.get("output", "")
        t = f"### Instruction:\n{instr}\n### Input:\n{inp}\n### Response:\n{out}"
        if len(t.strip()) >= 50:
            texts.append(t)
    return texts


def load_chat_data(split="train", max_samples=3000):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=False)
    texts = []
    for i, r in enumerate(ds):
        if i >= max_samples:
            break
        msgs = r.get("messages", [])
        t = ""
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            t += f"<|{role}|>\n{content}\n"
        t += "<|end|>"
        if len(t.strip()) >= 50:
            texts.append(t)
    return texts


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=256):
        self.seq_len = seq_len
        self.data = []

        for text in texts:
            ids = tokenizer.encode_prompt(text)
            self.data.extend(ids)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_len + 1]
        return (torch.tensor(chunk[:-1], dtype=torch.long),
                torch.tensor(chunk[1:], dtype=torch.long))


def evaluate(model, loader, criterion, device, max_batches=200):
    model.eval()
    total, count = 0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x, n_loops=4)
            loss = criterion(logits.view(-1, model.cfg.vocab_size), y.view(-1))
            total += loss.item() * x.size(0)
            count += x.size(0)
    return total / count


def train(cfg, args, device, tokenizer, train_loader, val_loader):
    model = TinyModel(cfg).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * 1000)

    start_epoch = 0
    best_loss = float("inf")
    global_step = 0

    latest_path = args.save_dir / "latest.pt"
    best_path = args.save_dir / "best.pt"

    def _sd(m, opt, sch, ep, st, bl):
        return {
            "model_state": m.state_dict(),
            "optimizer": opt.state_dict(),
            "sch": sch.state_dict() if sch else None,
            "epoch": ep, "step": st, "best_loss": bl,
            "cfg": {k: getattr(cfg, k) for k in
                    ["vocab_size", "dim", "n_heads", "max_seq_len", "max_loop_iters",
                     "prelude_layers", "coda_layers", "act_threshold", "rope_theta",
                     "lora_rank", "dropout", "tie_weights"]},
        }

    if args.resume and latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["sch"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"Resumed at epoch {start_epoch}, step {global_step}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x, n_loops=4)
            loss = criterion(logits.view(-1, model.cfg.vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            global_step += 1

            if global_step % args.save_every == 0:
                torch.save(_sd(model, optimizer, scheduler, epoch, global_step, best_loss), latest_path)
                val_loss = evaluate(model, val_loader, criterion, device)
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save({"model_state": model.state_dict(), "cfg": {
                        k: getattr(cfg, k) for k in cfg.__dataclass_fields__}}, best_path)
                    print(f"  New best! val_loss={val_loss:.4f}")
                model.train()

            if global_step % args.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"E{epoch+1} S{global_step} loss={loss.item():.4f} lr={lr:.6f}")

            if global_step >= args.max_steps > 0:
                break

        avg_loss = total_loss / max(1, batch_idx + 1)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs} train_loss={avg_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model_state": model.state_dict(), "cfg": {
                k: getattr(cfg, k) for k in cfg.__dataclass_fields__}}, best_path)
        torch.save(_sd(model, optimizer, scheduler, epoch, global_step, best_loss), latest_path)

        if global_step >= args.max_steps > 0:
            break

    return model


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wiki", type=int, default=3000)
    parser.add_argument("--code", type=int, default=2000)
    parser.add_argument("--chat", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.save_dir = Path(args.save_dir)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = TinyConfig()
    model = TinyModel(cfg)
    print(f"Params: {model.num_parameters():,} (vocab={cfg.vocab_size})")

    tokenizer = BPETokenizer(vocab_size=cfg.vocab_size)

    if os.path.exists(tokenizer.path):
        print(f"Loading existing BPE tokenizer from {tokenizer.path}")
    else:
        print("Training BPE tokenizer...")
        wiki = load_wikitext("train")[:5000]
        code = load_code_alpaca("train", max_samples=2000)
        chat = load_chat_data("train", max_samples=1000)
        all_texts = wiki + code + chat
        tokenizer.train(all_texts[:5000])

    print("Loading data...")
    wiki = load_wikitext("train")[:args.wiki]
    code = load_code_alpaca("train", max_samples=args.code)
    chat = load_chat_data("train", max_samples=args.chat)
    wiki_v = load_wikitext("validation")[:50]

    train_texts = wiki + code + chat
    val_texts = wiki_v
    print(f"Train: {len(train_texts)} texts, Val: {len(val_texts)} texts")

    train_ds = TextDataset(train_texts, tokenizer, args.seq_len)
    val_ds = TextDataset(val_texts, tokenizer, args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = train(cfg, args, device, tokenizer, train_loader, val_loader)

    ckpt = torch.load(args.save_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    for prompt in ["def fibonacci", "The meaning of life is", "Hello, how are"]:
        ids = torch.tensor([tokenizer.encode_prompt(prompt)]).to(device)
        out = model.generate(ids, max_new_tokens=40, n_loops=4, temperature=0.7)
        print(f"Prompt: {prompt}")
        print(f"  Gen: {tokenizer.decode(out[0].tolist())}\n")


if __name__ == "__main__":
    main()