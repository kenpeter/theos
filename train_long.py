"""
Long-run training: streams billions of tokens from FineWeb + wiki + code.
Resumable, checkpoints every N steps. Run for 100K+ steps.
"""

import os, torch, torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from tiny_model import TinyConfig, TinyModel
from bpe_tokenizer import BPETokenizer


def multi_stream():
    """Infinite interleaved stream from FineWeb + wiki + code."""
    from datasets import load_dataset, interleave_datasets

    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train", streaming=True)
    code = load_dataset("sahil2801/CodeAlpaca-20k", split="train", streaming=True)
    web = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)

    def wiki_gen():
        for r in wiki:
            t = r["text"].strip()
            if len(t) >= 50:
                yield t

    def code_gen():
        for r in code:
            parts = [r.get(k, "") for k in ["instruction", "input", "output"]]
            t = f"<|user|>\n{parts[0]}\n{parts[1]}\n<|assistant|>\n{parts[2]}\n<|end|>\n"
            if len(t) >= 50:
                yield t

    def web_gen():
        for r in web:
            t = r.get("text", "")
            if len(t) >= 100:
                yield t

    while True:
        for t in wiki_gen(): yield t
        for t in code_gen(): yield t
        for _ in range(5): 
            for t in web_gen(): yield t


class TextStream(IterableDataset):
    def __init__(self, tokenizer, seq_len=256):
        self.tok = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        buf = []
        for text in multi_stream():
            ids = self.tok.encode_prompt(text)
            buf.extend(ids)
            while len(buf) >= self.seq_len + 1:
                yield (torch.tensor(buf[:self.seq_len], dtype=torch.long),
                       torch.tensor(buf[1:self.seq_len + 1], dtype=torch.long))
                buf = buf[self.seq_len:]


def evaluate(model, loader, criterion, device, steps=100):
    model.eval()
    total, count = 0.0, 0
    it = iter(loader)
    with torch.no_grad():
        for _ in range(steps):
            x, y = next(it)
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x, n_loops=4).view(-1, model.cfg.vocab_size), y.view(-1))
            total += loss.item() * x.size(0)
            count += x.size(0)
    return total / count


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bpe_path", type=str, default="bpe_tokenizer_8k.json")
    args = parser.parse_args()

    args.save_dir = Path(args.save_dir)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = TinyConfig()
    model = TinyModel(cfg).to(device)
    tok = BPETokenizer(vocab_size=cfg.vocab_size, path=args.bpe_path)
    print(f"Params: {model.num_parameters():,}")

    ds = TextStream(tok, args.seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_steps)

    step = 0
    best_loss = float("inf")
    latest_path = args.save_dir / "latest.pt"
    best_path = args.save_dir / "best.pt"

    if args.resume and latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer"])
        sch.load_state_dict(ckpt["sch"])
        step = ckpt["step"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"Resumed at step {step}")

    model.train()
    data_iter = iter(dl)

    while step < args.max_steps:
        x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        logits = model(x, n_loops=4)
        loss = criterion(logits.view(-1, cfg.vocab_size), y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        opt.zero_grad()

        if step % args.log_every == 0:
            lr = opt.param_groups[0]["lr"]
            print(f"S{step} loss={loss.item():.4f} lr={lr:.6f}")

        if step > 0 and step % args.save_every == 0:
            sd = {
                "model_state": model.state_dict(),
                "optimizer": opt.state_dict(),
                "sch": sch.state_dict(),
                "step": step,
                "best_loss": best_loss,
            }
            torch.save(sd, latest_path)
            val_loss = evaluate(model, dl, criterion, device, 200)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({"model_state": model.state_dict()}, best_path)
                print(f"  ★ New best! val_loss={val_loss:.4f}")
            else:
                print(f"  val_loss={val_loss:.4f} best={best_loss:.4f}")
            model.train()

        step += 1

    print(f"Done {step} steps. Best val_loss={best_loss:.4f}")


if __name__ == "__main__":
    main()