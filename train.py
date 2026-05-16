"""
Multi-stage training with WSD scheduler.
Stage 1: general + broad code (warmup + stable)
Stage 2: add reasoning + specialized code (stable)
Stage 3: highest quality only + annealing (decay LR to 0)
"""

import os, pickle, time, torch, torch.nn as nn, numpy as np
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

    MAX_TEXT_CHARS = 20000

    def tokenize_texts(self, texts, cache_path, prog_name=""):
        if cache_path and os.path.exists(cache_path):
            arr = np.load(cache_path)
            return arr
        flat = []
        start = time.time()
        skipped = 0
        step = max(1, len(texts) // 8)
        for i, t in enumerate(texts):
            if len(t) > self.MAX_TEXT_CHARS or "\x00" in t:
                skipped += 1
                continue
            flat.extend(self.encode_prompt(t))
            if prog_name and step > 0 and (i+1) % step == 0:
                print(f"  tokenizing {prog_name}: {i+1}/{len(texts)} ({len(flat):,} tok, {time.time()-start:.0f}s, skipped={skipped})")
        if skipped:
            print(f"  skipped {skipped} texts over {self.MAX_TEXT_CHARS} chars or with null bytes")
        arr = np.array(flat, dtype=np.int32)
        if cache_path:
            np.save(cache_path, arr)
        return arr


# ── Dataset ────────────────────────────────────────────────────────

class FlatDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.seq_len = seq_len
        if isinstance(tokens, np.ndarray):
            self.tokens = torch.from_numpy(tokens).long()
        else:
            self.tokens = torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, i):
        x = self.tokens[i : i + self.seq_len]
        y = self.tokens[i + 1 : i + self.seq_len + 1]
        return x.clone(), y.clone()


# ── WSD Scheduler ──────────────────────────────────────────────────

class WSDScheduler:
    def __init__(self, optimizer, warmup_steps=200, peak_lr=3e-4):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self._step = 0
        self._mode = "warmup"
        self._decay_start = None
        self._decay_steps = None
        self._set_lr(0.0)

    def _set_lr(self, lr):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def step(self):
        self._step += 1
        if self._mode == "warmup":
            progress = min(1.0, self._step / max(1, self.warmup_steps))
            self._set_lr(self.peak_lr * progress)
            if self._step >= self.warmup_steps:
                self._mode = "stable"
        elif self._mode == "stable":
            self._set_lr(self.peak_lr)
        elif self._mode == "decay":
            elapsed = self._step - self._decay_start
            progress = min(1.0, elapsed / max(1, self._decay_steps))
            self._set_lr(self.peak_lr * max(0.0, 1.0 - progress))

    def begin_decay(self, decay_steps):
        self._mode = "decay"
        self._decay_start = self._step
        self._decay_steps = decay_steps
        print(f"  Annealing started: LR decays to 0 over {decay_steps} steps")


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
    ("web", ("HuggingFaceFW/fineweb", "sample-10BT"), True, 50000, None),
    ("oa", ("timdettmers/openassistant-guanaco",), True, 15000, None),
    ("magicoder", ("ise-uiuc/Magicoder-Evol-Instruct-110K",), True, 50000,
     lambda r, _: f"<|user|>\n{r['instruction']}\n<|assistant|>\n{r['response']}\n<|end|>"),
]

LEETCODE_BASE = Path("/home/kenpeter/work/data")
DATA_CACHE = Path("/home/kenpeter/work/data/_cache")

def _fmt(problem, reasoning="", code=""):
    if reasoning and code:
        return f"<|user|>\n{problem}\n<|reasoning|>\n{reasoning}\n<|assistant|>\n{code}\n<|end|>"
    elif code:
        return f"<|user|>\n{problem}\n<|assistant|>\n{code}\n<|end|>"
    else:
        return f"<|user|>\n{problem}\n<|assistant|>\nI'd be happy to help solve this problem.\n<|end|>"

def _load_all_leetcode():
    """Load ALL LeetCode datasets from /home/kenpeter/work/data/."""
    import json, pandas as pd
    cache = CACHE / "all_leetcode.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    texts = []

    # 1. high_quality_leetcode (2,638 rows) — problem + CoT + code
    hq_path = LEETCODE_BASE / "high_quality_leetcode" / "train.jsonl"
    if hq_path.exists():
        with open(hq_path) as f:
            for line in f:
                d = json.loads(line)
                texts.append(_fmt(d["problem_description"], d["high_quality_cot"], d["completion"]))
        print(f"  ✓ high_quality_leetcode: {sum(1 for _ in open(hq_path))}")

    # 2. LeetCode_YT_CC_CoT_Summary (17,053 rows) — problem + summary CoT + python code
    yt_dir = LEETCODE_BASE / "LeetCode_YT_CC_CoT_Summary" / "data"
    for fn in sorted(yt_dir.glob("*.parquet")):
        df = pd.read_parquet(fn)
        for _, r in df.iterrows():
            texts.append(_fmt(r["question_content"], r.get("Summary", ""), r.get("python", "")))
        print(f"  ✓ LeetCode_YT_CC_CoT_Summary ({fn.name}): {len(df)}")

    # 3. DenCT_LeetCode (19,011 rows) — problem + explanation (code is embedded in content)
    denct_path = LEETCODE_BASE / "DenCT_LeetCode" / "leetcode-java-python.parquet"
    if denct_path.exists():
        df = pd.read_parquet(denct_path)
        for _, r in df.iterrows():
            problem = r.get("question_content", "")
            reasoning = r.get("content", "")
            texts.append(_fmt(problem, reasoning, ""))
        print(f"  ✓ DenCT_LeetCode: {len(df)}")

    # 4. LimYeri_LeetCode (15,734 rows) — conversations array with system/user/assistant
    lim_path = LEETCODE_BASE / "LimYeri_LeetCode" / "train.parquet"
    if lim_path.exists():
        df = pd.read_parquet(lim_path)
        count = 0
        for _, r in df.iterrows():
            convs = r.get("conversations", [])
            problem, reasoning, code = "", "", ""
            for msg in convs:
                if isinstance(msg, dict):
                    role = msg.get("from", "")
                    val = msg.get("value", "")
                    if role == "user":
                        problem = val
                    elif role == "assistant":
                        if "```" in val:
                            parts = val.split("```")
                            reasoning = parts[0].strip()
                            code = "```" + "```".join(parts[1:]) if len(parts) > 1 else ""
                        else:
                            reasoning = val
            if problem:
                texts.append(_fmt(problem, reasoning, code))
                count += 1
        print(f"  ✓ LimYeri_LeetCode: {count}")

    # 5. mesolitica_LeetCodeQwQ (1,561 rows) — problem + QwQ reasoning + code
    qwq_path = LEETCODE_BASE / "mesolitica_LeetCodeQwQ" / "train.parquet"
    if qwq_path.exists():
        df = pd.read_parquet(qwq_path)
        for _, r in df.iterrows():
            texts.append(_fmt(r["content"], r.get("qwq", ""), r.get("solution", "")))
        print(f"  ✓ mesolitica_LeetCodeQwQ: {len(df)}")

    # 6. newfacade_LeetCodeDataset (2,869 rows) — problem + response reasoning + completion code
    nf_dir = LEETCODE_BASE / "newfacade_LeetCodeDataset"
    for fn in ["leetcode_train.jsonl", "leetcode_test.jsonl"]:
        p = nf_dir / fn
        if p.exists():
            with open(p) as f:
                count = 0
                for line in f:
                    d = json.loads(line)
                    texts.append(_fmt(d["problem_description"], d.get("response", ""), d.get("completion", "")))
                    count += 1
            print(f"  ✓ newfacade_LeetCodeDataset ({fn}): {count}")

    # 7. greengerong_LeetCode (2,360 rows) — problem + python (explanation + code in one field)
    gg_path = LEETCODE_BASE / "greengerong_LeetCode" / "leetcode-train.jsonl"
    if gg_path.exists():
        with open(gg_path) as f:
            count = 0
            for line in f:
                d = json.loads(line)
                py_code = d.get("python", "")
                texts.append(_fmt(d["content"], "", py_code))
                count += 1
        print(f"  ✓ greengerong_LeetCode: {count}")

    # 8. juyoungml_LeetCodeRosetta (2,359 rows) — problem + python code (no reasoning)
    jr_path = LEETCODE_BASE / "juyoungml_LeetCodeRosetta" / "data" / "train-00000-of-00001.parquet"
    if jr_path.exists():
        df = pd.read_parquet(jr_path)
        for _, r in df.iterrows():
            texts.append(_fmt(r["content"], "", r.get("python_code", "")))
        print(f"  ✓ juyoungml_LeetCodeRosetta: {len(df)}")

    # 9. vovw_LeetCode (2,360 rows) — problem + python code (no reasoning)
    vovw_path = LEETCODE_BASE / "vovw_LeetCode" / "dataset.parquet"
    if vovw_path.exists():
        df = pd.read_parquet(vovw_path)
        for _, r in df.iterrows():
            texts.append(_fmt(r["content"], "", r.get("python", "")))
        print(f"  ✓ vovw_LeetCode: {len(df)}")

    pickle.dump(texts, open(cache, "wb"))
    print(f"  Total All LeetCode: {len(texts)} texts loaded")
    return texts


def _load_roleplay():
    """Load bluemoon_roleplay data as general dialogue text."""
    import pyarrow as pa
    cache = CACHE / "roleplay.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    arrow_path = LEETCODE_BASE / "bluemoon_roleplay" / "train" / "data-00000-of-00001.arrow"
    if not arrow_path.exists():
        print("  ⚠ roleplay not found, skipping")
        return []
    with pa.memory_map(str(arrow_path), "rb") as mm:
        reader = pa.ipc.RecordBatchStreamReader(mm)
        table = reader.read_all()
    texts = []
    for i in range(len(table)):
        msg = table.column("message")[i].as_py()
        if msg and len(msg) >= 100:
            texts.append(f"<|user|>\nContinue the story.\n<|assistant|>\n{msg}\n<|end|>")
    pickle.dump(texts, open(cache, "wb"))
    print(f"  ✓ bluemoon_roleplay: {len(texts)} texts")
    return texts


def _load_code_datasets():
    """Load new code datasets from /home/kenpeter/work/data/_cache/."""
    code_files = ["codeparrot_clean", "self_oss_instruct", "codefeedback", "dolphin_coder", "smoltalk"]
    max_per_dataset = {"smoltalk": 50000}
    train_all, val_all = [], []
    for name in code_files:
        path = DATA_CACHE / f"{name}.pkl"
        if not path.exists():
            print(f"  ⚠ {name}.pkl not found, skipping")
            continue
        texts = pickle.load(open(path, "rb"))
        limit = max_per_dataset.get(name, len(texts))
        if limit < len(texts):
            texts = texts[:limit]
            print(f"  trimmed {name} to {limit}")
        split = max(1, len(texts) // 20)
        if split > 10000:
            split = 10000
        train_all.extend(texts[:-split])
        val_all.extend(texts[-split:])
        print(f"  ✓ {name}: {len(texts)} texts")
    return train_all, val_all


# ── Evaluation ────────────────────────────────────────────────────

def evaluate(model, loader, criterion, device, max_batches=0):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches: break
            x, y = x.to(device), y.to(device)
            logits, _ = model(x, n_loops=4)
            loss = criterion(logits.view(-1, model.cfg.vocab_size), y.view(-1))
            total += loss.item() * x.size(0)
            count += x.size(0)
    return total / count if count > 0 else 0.0


class MetricsTracker:
    def __init__(self, path):
        self.path = path
        self.history = []
        if os.path.exists(path):
            try:
                import json
                self.history = json.load(open(path))
            except: pass

    def log(self, step, train_loss, val_loss, code_acc, lr, best_loss):
        import json
        import datetime
        entry = {
            "step": step, "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4) if val_loss else None,
            "code_acc": round(code_acc, 4) if code_acc else None,
            "lr": lr, "best_loss": round(best_loss, 4),
            "time": datetime.datetime.now().isoformat(),
        }
        self.history.append(entry)
        json.dump(self.history, open(self.path, "w"), indent=2)

    def stall_count(self, key="val_loss", patience=3):
        recent = [e[key] for e in self.history[-patience:] if e.get(key) is not None]
        if len(recent) < patience:
            return 0
        return sum(1 for i in range(1, len(recent)) if recent[i] >= recent[i-1])


def eval_code_syntax(model, tok, device, n_samples=4):
    prompts = [
        "<|user|>\nWrite a Python function that returns the nth Fibonacci number.\n<|assistant|>\ndef fibonacci(n):",
        "<|user|>\nWrite a Python function that sorts an array.\n<|assistant|>\ndef sort_array(arr):",
        "<|user|>\nWrite a Python class implementing a stack.\n<|assistant|>\nclass Stack:",
        "<|user|>\nWrite a Python function that performs binary search.\n<|assistant|>\ndef binary_search(arr, target):",
    ]
    model.eval()
    score = 0.0
    for prompt in prompts[:n_samples]:
        ids = torch.tensor([tok.encode_prompt(prompt)]).to(device)
        out = model.generate(ids, max_new_tokens=60, n_loops=4, temperature=0.3, top_k=40)
        text = tok.decode(out[0].tolist())
        try:
            compile(text, "<eval>", "exec")
            score += 1.0
        except SyntaxError:
            pass
    return score / max(1, n_samples)


# ── Main ───────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint (auto-detected if exists)")
    parser.add_argument("--fresh", action="store_true", help="Force fresh training, ignore checkpoints")
    parser.add_argument("--data_only", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--grad_acc", type=int, default=1)
    parser.add_argument("--micro", action="store_true")
    parser.add_argument("--cooldown", type=float, default=0.0)
    parser.add_argument("--aux_scale", type=float, default=0.01)
    parser.add_argument("--synth_only", action="store_true")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--pin_memory", action="store_true", default=True, help="Pin memory for DataLoader")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for model optimization")
    parser.add_argument("--lr_patience", type=int, default=3, help="Evals without improvement before LR decay")
    parser.add_argument("--lr_decay", type=float, default=0.5, help="LR decay factor on plateau")
    parser.add_argument("--metrics_file", type=str, default="training_metrics.json", help="Metrics log path")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--label_smooth", type=float, default=0.1, help="Label smoothing epsilon")
    # Stage controls
    parser.add_argument("--stage1_pct", type=float, default=0.60, help="% of steps for stage 1")
    parser.add_argument("--stage2_pct", type=float, default=0.25, help="% of steps for stage 2")
    args = parser.parse_args()

    if args.micro:
        args.max_steps = 30
        args.save_every = 10
        args.log_every = 5
        args.batch_size = 2
        args.seq_len = 64
        args.eval_steps = 5
        args.patience = 2
        print("Micro mode: 30 steps")
    elif args.quick:
        args.max_steps = 300
        args.save_every = 50
        args.log_every = 25
        args.eval_steps = 20
        args.patience = 2
        print("Quick mode: 300 steps")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # ── Tokenizer ──
    tok = BPETok(8192, "tokenizer.json")
    if not os.path.exists("tokenizer.json") and os.path.exists("bpe_tokenizer_8k.json"):
        os.rename("bpe_tokenizer_8k.json", "tokenizer.json")
        tok = BPETok(8192, "tokenizer.json")
    if not os.path.exists("tokenizer.json"):
        print("Training tokenizer...")
        sample = _load_or_download("wiki", ("Salesforce/wikitext", "wikitext-2-v1"), False, 0, None)[:3000]
        sample += _load_all_leetcode()[:500]
        tok.train(sample)

    # ── Data ──
    syn_path = CACHE / "synthetic_code_reasoning.pkl"
    all_texts, val_texts = [], []

    if args.synth_only:
        print("Synth-only mode")
        syn = pickle.load(open(syn_path, "rb"))
        split = max(1, len(syn) // 20)
        all_texts = syn[:-split]
        val_texts = syn[-split:]
    else:
        for name, url, streaming, max_s, fn in DATASETS:
            _load_or_download(name, url, streaming, max_s, fn)
        for name, _, _, _, _ in DATASETS:
            texts = pickle.load(open(CACHE / f"{name}.pkl", "rb"))
            if name == "wiki":
                val_texts.extend(texts[-200:])
                texts = texts[:-200]
            all_texts.extend(texts)

        leetcode = _load_all_leetcode()
        split = max(1, len(leetcode) // 20)
        if split > 5000: split = 5000
        all_texts.extend(leetcode[:-split])
        val_texts.extend(leetcode[-split:])

        if syn_path.exists():
            syn = pickle.load(open(syn_path, "rb"))
            split = max(1, len(syn) // 20)
            all_texts.extend(syn[:-split])
            val_texts.extend(syn[-split:])

        # Roleplay data
        roleplay = _load_roleplay()
        split = max(1, len(roleplay) // 20)
        all_texts.extend(roleplay[:-split])
        val_texts.extend(roleplay[-split:])

        # Harvested LeetCode data
        harvest_path = CACHE / "harvested_data.pkl"
        if harvest_path.exists():
            harvest = pickle.load(open(harvest_path, "rb"))
            split = max(1, len(harvest) // 20)
            all_texts.extend(harvest[:-split])
            val_texts.extend(harvest[-split:])
            print(f"  ✓ harvested LeetCode: {len(harvest)} texts")

        # New code datasets
        code_train, code_val = _load_code_datasets()
        all_texts.extend(code_train)
        val_texts.extend(code_val)

    print(f"Total texts: {len(all_texts)} train + {len(val_texts)} val")

    # Tokenize
    tok_cache = CACHE / "token_cache"
    tok_cache.mkdir(exist_ok=True)
    train_cache = str(tok_cache / "train.npy")
    val_cache = str(tok_cache / "val.npy")
    flat_train = tok.tokenize_texts(all_texts, train_cache, "train")
    flat_val = tok.tokenize_texts(val_texts, val_cache, "val")
    print(f"Tokens: {len(flat_train):,} train, {len(flat_val):,} val")

    if args.data_only:
        print("Data ready.")
        return

    # ── Model ──
    cfg = TinyConfig(dropout=args.dropout)
    model = TinyModel(cfg).to(device)
    print(f"Params: {model.num_parameters():,}")

    # ── Smoke test: verify generate() produces real tokens ──
    _smoke_ids = torch.tensor([tok.encode_prompt("def hello(")]).to(device)
    with torch.no_grad():
        _smoke_out = model.generate(_smoke_ids, max_new_tokens=10, n_loops=4, temperature=0.3)
    _smoke_decoded = tok.decode(_smoke_out[0].tolist())
    if len(_smoke_decoded) <= len("def hello("):
        print("SMOKE FAIL: generate() produces no new tokens. Aborting.", flush=True)
        return
    _unique_chars = len(set(_smoke_decoded.strip()))
    if _unique_chars <= 3:
        print(f"SMOKE WARN: generate() output is degenerate ({_unique_chars} unique chars). Proceed with caution.", flush=True)
    else:
        print(f"SMOKE OK: generate() produces {_unique_chars} unique chars.", flush=True)

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("Compiled with torch.compile")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = WSDScheduler(opt, warmup_steps=args.warmup, peak_lr=args.lr)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    scaler = None  # bfloat16 doesn't need GradScaler
    metrics = MetricsTracker(args.metrics_file)
    lr_stall = 0

    # Validation loader
    val_ds = FlatDataset(flat_val, args.seq_len)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, persistent_workers=args.num_workers > 0)

    # Stage boundaries
    stage1_end = int(args.max_steps * args.stage1_pct)
    stage2_end = stage1_end + int(args.max_steps * args.stage2_pct)

    # ── Resume (auto-detects if checkpoint exists) ──
    step, best_loss = 0, float("inf")
    has_ckpt = os.path.exists("checkpoints/latest.pt")
    should_resume = (args.resume or has_ckpt) and not args.fresh
    if should_resume and has_ckpt:
        ckpt = torch.load("checkpoints/latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer"])
        sched._step = ckpt.get("step", 0)
        step = ckpt.get("step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        sched._mode = ckpt.get("sched_mode", "warmup")
        lr_stall = ckpt.get("lr_stall", 0)
        sched.peak_lr = ckpt.get("peak_lr", args.lr)
        print(f"Resumed at step {step} (LR={sched.peak_lr:.2e})", flush=True)
    elif args.fresh and has_ckpt:
        os.remove("checkpoints/latest.pt")
        if os.path.exists("checkpoints/best.pt"):
            os.remove("checkpoints/best.pt")
        print("Fresh start — cleared checkpoints", flush=True)

    os.makedirs("checkpoints", exist_ok=True)
    model.train()
    opt.zero_grad()
    stall = 0

    # Create training dataloader (use infinite sampler to avoid huge randperm)
    train_ds = FlatDataset(flat_train, args.seq_len)

    class InfiniteSampler(torch.utils.data.Sampler):
        def __init__(self, n):
            self.n = n
        def __iter__(self):
            while True:
                yield from (int(i) for i in torch.randint(0, self.n, (self.n // 1000 + 1,)))
        def __len__(self):
            return self.n

    sampler = InfiniteSampler(len(train_ds))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=True, persistent_workers=args.num_workers > 0)
    train_iter = iter(train_dl)
    current_stage = 0  # 0=stage1, 1=stage2, 2=stage3
    t0 = time.time()

    for step in range(step, args.max_steps):
        # Check stage transition
        new_stage = 0
        if step >= stage2_end:
            new_stage = 2
        elif step >= stage1_end:
            new_stage = 1

        if new_stage != current_stage:
            current_stage = new_stage
            stage_names = ["Stage 1 — General + Broad Code",
                           "Stage 2 — Reasoning + Specialized Code",
                           "Stage 3 — High Quality + Annealing"]
            print(f"\n{'='*50}")
            print(f"Entering {stage_names[current_stage]}")
            if current_stage == 2:
                decay_steps = args.max_steps - step
                sched.begin_decay(decay_steps)
            # Re-shuffle data for new stage
            train_iter = iter(train_dl)
            print(f"{'='*50}\n")

        # Training step
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None, dtype=torch.bfloat16):
            logits, aux_loss = model(x, n_loops=4)
            nll = crit(logits.view(-1, cfg.vocab_size), y.view(-1))
            loss = nll + args.aux_scale * aux_loss
            loss = loss / args.grad_acc

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % args.grad_acc == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            sched.step()
            opt.zero_grad()
        nll_val = nll.item()

        if args.cooldown > 0:
            time.sleep(args.cooldown)

        if step % args.log_every == 0:
            lr = opt.param_groups[0]["lr"]
            stage_names = ["S1", "S2", "S3"]
            elapsed = time.time() - t0
            print(f"S{step} [{stage_names[current_stage]}] loss={nll_val:.4f} lr={lr:.2e} t={elapsed:.0f}s")

        if step > 0 and step % args.log_every == 0:
            torch.save({
                "model_state": model.state_dict(),
                "optimizer": opt.state_dict(),
                "sched_mode": sched._mode,
                "step": step, "best_loss": best_loss,
                "lr_stall": lr_stall, "peak_lr": sched.peak_lr,
            }, "checkpoints/latest.pt")

        if step > 0 and step % args.save_every == 0:
            v = evaluate(model, val_dl, crit, device, args.eval_steps)
            code_acc = eval_code_syntax(model, tok, device)
            metrics.log(step, nll_val, v, code_acc, opt.param_groups[0]["lr"], best_loss)
            improved = v < best_loss
            if improved:
                best_loss = v
                torch.save({"model_state": model.state_dict()}, "checkpoints/best.pt")
                print(f"  ★ Best val_loss={v:.4f} code_acc={code_acc:.2f}")
                stall = 0
                lr_stall = 0
            else:
                stall += 1
                lr_stall += 1
                print(f"  val_loss={v:.4f} code_acc={code_acc:.2f} best={best_loss:.4f} stall={stall}/{args.patience}")
                # Adaptive LR decay on plateau (only during stable stage, not during planned decay)
                if lr_stall >= args.lr_patience and sched._mode == "stable":
                    new_lr = opt.param_groups[0]["lr"] * args.lr_decay
                    sched.peak_lr = new_lr
                    sched._set_lr(new_lr)
                    lr_stall = 0
                    print(f"  ◆ Adaptive LR decay to {new_lr:.2e} (plateau x{args.lr_patience})")
            torch.save({
                "model_state": model.state_dict(),
                "optimizer": opt.state_dict(),
                "sched_mode": sched._mode,
                "step": step, "best_loss": best_loss,
                "lr_stall": lr_stall, "peak_lr": sched.peak_lr,
            }, "checkpoints/latest.pt")
            model.train()

            # ── Eval Gate: real code evaluation ──
            if step >= 10000:
                import subprocess as _sp
                try:
                    _eg = _sp.run(
                        ["python3", "eval_real.py", "checkpoints/best.pt"],
                        capture_output=True, text=True, timeout=120,
                    )
                    _passes = _eg.stdout.count("[PASS]")
                    _fails = _eg.stdout.count("[FAIL]")
                    _total = _passes + _fails
                    _compile_ok = _eg.stdout.count("syntax error") + _eg.stdout.count("no valid def")
                    _compile_acc = max(0, (_total - _compile_ok)) / max(1, _total)
                    print(f"  ╔══════════════════════════════════════╗", flush=True)
                    print(f"  ║   EVAL GATE — Step {step}", flush=True)
                    print(f"  ║   tests: {_passes}/{_total}  |  compile: {_compile_acc:.0%}", flush=True)
                    status = "GREEN" if _passes > 0 else ("YELLOW" if _compile_acc >= 0.10 else "RED")
                    print(f"  ║   status: {status}", flush=True)
                    print(f"  ╚══════════════════════════════════════╝", flush=True)
                    if step >= 30000 and _total > 0 and _compile_acc < 0.10:
                        print(f"  ███ EVAL GATE: HALTING — compile_acc={_compile_acc:.2f} at step {step}", flush=True)
                        break
                except Exception as _e:
                    print(f"  Eval Gate: skipped ({_e})", flush=True)
            if stall >= args.patience:
                print(f"  Early stopping")
                break

    print(f"\nDone {step+1} steps. Best val_loss={best_loss:.4f}")

    # Quick generation test
    model.eval()
    for prompt in ["def fibonacci", "The meaning of life"]:
        ids = torch.tensor([tok.encode_prompt(prompt)]).to(device)
        out = model.generate(ids, max_new_tokens=30, n_loops=4, temperature=0.7)
        print(f"  Gen: {tok.decode(out[0].tolist())}")


if __name__ == "__main__":
    main()
