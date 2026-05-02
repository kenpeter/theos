"""
Consolidated training — downloads data, tokenizes, trains. All in one.
"""

import os, pickle, torch, torch.nn as nn, numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tiny_model import TinyConfig, TinyModel


# ── BPE Tokenizer ──────────────────────────────────────────────────

class BPETok:
    SPECIALS = ["<pad>", "<eos>", "<unk>", "<bos>"]
    PAD, EOS, UNK, BOS = 0, 1, 2, 3

    def __init__(self, vocab_size=8192, path="tokenizer.json"):
        self.vocab_size = vocab_size
        self.path = path
        if os.path.exists(path):
            self.tk = Tokenizer.from_file(path)
        else:
            self.tk = Tokenizer(models.BPE(unk_token="<unk>"))
            self.tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self.tk.decoder = decoders.ByteLevel()
            self.tk.post_processor = processors.ByteLevel(trim_offsets=False)

    def train(self, texts):
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size, special_tokens=self.SPECIALS,
            show_progress=True, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        self.tk.train_from_iterator(texts, trainer)
        self.tk.save(self.path)

    def encode(self, text): return [self.BOS] + self.tk.encode(text).ids + [self.EOS]
    def encode_prompt(self, text): return self.tk.encode(text).ids
    def decode(self, ids):
        ids = [i for i in ids if i not in (self.PAD, self.BOS)]
        if self.EOS in ids: ids = ids[:ids.index(self.EOS)]
        return self.tk.decode(ids)

    def tokenize_texts(self, texts, cache_path):
        if cache_path and os.path.exists(cache_path):
            return np.load(cache_path).tolist()
        flat = []
        for i, t in enumerate(texts):
            flat.extend(self.encode_prompt(t))
        if cache_path:
            np.save(cache_path, np.array(flat, dtype=np.int32))
        return flat


# ── Dataset ────────────────────────────────────────────────────────

class FlatDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens, self.seq_len = tokens, seq_len
    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)
    def __getitem__(self, i):
        return (torch.tensor(self.tokens[i:i+self.seq_len], dtype=torch.long),
                torch.tensor(self.tokens[i+1:i+self.seq_len+1], dtype=torch.long))


# ── Data Loading ───────────────────────────────────────────────────

CACHE = Path("dataset_cache")
CACHE.mkdir(exist_ok=True)

def _load_or_download(name, url, streaming=False, max_samples=0, map_fn=None):
    path = CACHE / f"{name}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    print(f"Downloading {name}...")
    from datasets import load_dataset
    ds = load_dataset(*url, split="train", streaming=streaming)
    texts = []
    for i, r in enumerate(ds):
        if max_samples and i >= max_samples: break
        t = r.get("text", "") if isinstance(r, dict) else str(r)
        if map_fn: t = map_fn(r, t)
        if len(t.strip()) >= 50: texts.append(t)
    pickle.dump(texts, open(path, "wb"))
    print(f"  {len(texts)} texts saved")
    return texts

DATASETS = [
    ("wiki", ("Salesforce/wikitext", "wikitext-2-v1"), False, 0, None),
    ("code", ("sahil2801/CodeAlpaca-20k",), False, 0,
     lambda r, _: f"<|user|>\n{r['instruction']}\n### Input:\n{r['input']}\n<|assistant|>\n{r['output']}\n<|end|>"),
    ("web", ("HuggingFaceFW/fineweb", "sample-10BT"), True, 100000, None),
    ("chat", ("timdettmers/openassistant-guanaco",), True, 15000, None),
]


# ── Training ───────────────────────────────────────────────────────

def evaluate(model, loader, criterion, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x, n_loops=4).view(-1, model.cfg.vocab_size), y.view(-1))
            total += loss.item() * x.size(0)
            count += x.size(0)
    return total / count


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data_only", action="store_true", help="download & tokenize only, no training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Tokenizer ──
    tok = BPETok(8192, "tokenizer.json")
    # Reuse old tokenizer if exists
    if not os.path.exists("tokenizer.json") and os.path.exists("bpe_tokenizer_8k.json"):
        os.rename("bpe_tokenizer_8k.json", "tokenizer.json")
        tok = BPETok(8192, "tokenizer.json")
    if not os.path.exists("tokenizer.json"):
        print("Training tokenizer...")
        sample = _load_or_download("wiki", ("Salesforce/wikitext", "wikitext-2-v1"), False, 0, None)[:5000]
        tok.train(sample)

    # ── Data ──
    for name, url, streaming, max_s, fn in DATASETS:
        _load_or_download(name, url, streaming, max_s, fn)

    all_texts, val_texts = [], []
    for name, _, _, _, _ in DATASETS:
        texts = pickle.load(open(CACHE / f"{name}.pkl", "rb"))
        if name == "wiki":
            val_texts = texts[-200:]
            texts = texts[:-200]
        all_texts.extend(texts)

    print(f"Total texts: {len(all_texts)} training + {len(val_texts)} validation")

    train_cache = "dataset_cache/train_tokens.npy" if os.path.exists("dataset_cache/train_tokens.npy") else str(CACHE / "train_tokens.npy")
    val_cache = "dataset_cache/val_tokens.npy" if os.path.exists("dataset_cache/val_tokens.npy") else str(CACHE / "val_tokens.npy")
    flat_train = tok.tokenize_texts(all_texts, train_cache)
    flat_val = tok.tokenize_texts(val_texts, val_cache)
    print(f"Tokens: {len(flat_train):,} train, {len(flat_val):,} val")

    if args.data_only:
        print("Data ready. Run without --data_only to train.")
        return

    # ── Model ──
    cfg = TinyConfig()
    model = TinyModel(cfg).to(device)
    print(f"Params: {model.num_parameters():,}")

    ds = FlatDataset(flat_train, args.seq_len)
    vl = FlatDataset(flat_val, args.seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    vl_dl = DataLoader(vl, batch_size=args.batch_size, shuffle=False, num_workers=0)

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_steps)

    step, best_loss = 0, float("inf")
    if args.resume and os.path.exists("checkpoints/latest.pt"):
        ckpt = torch.load("checkpoints/latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer"])
        sch.load_state_dict(ckpt["scheduler"])
        step = ckpt.get("step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"Resumed at step {step}")
    elif os.path.exists("checkpoints/best.pt"):
        ckpt = torch.load("checkpoints/best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state", ckpt))
        print("Loaded best.pt, starting fresh scheduler")

    os.makedirs("checkpoints", exist_ok=True)
    model.train()

    while step < args.max_steps:
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            loss = crit(model(x, n_loops=4).view(-1, cfg.vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            step += 1

            if step % args.log_every == 0:
                print(f"S{step} loss={loss.item():.4f} lr={opt.param_groups[0]['lr']:.6f}")

            if step % args.save_every == 0:
                v = evaluate(model, vl_dl, crit, device)
                if v < best_loss:
                    best_loss = v
                    torch.save({"model_state": model.state_dict()}, "checkpoints/best.pt")
                    print(f"  ★ New best val_loss={v:.4f}")
                else:
                    print(f"  val_loss={v:.4f} best={best_loss:.4f}")
                torch.save({"model_state": model.state_dict(), "optimizer": opt.state_dict(),
                            "scheduler": sch.state_dict(), "step": step, "best_loss": best_loss},
                           "checkpoints/latest.pt")
                model.train()

            if step >= args.max_steps: break
        if step >= args.max_steps: break

    print(f"Done. Best val_loss={best_loss:.4f}")


if __name__ == "__main__":
    main()