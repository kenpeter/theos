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
        self.tokens = tokens
        self.seq_len = seq_len
    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)
    def __getitem__(self, i):
        return (torch.tensor(self.tokens[i:i+self.seq_len], dtype=torch.long),
                torch.tensor(self.tokens[i+1:i+self.seq_len+1], dtype=torch.long))


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


def eval_code_syntax(model, tok, device, n_samples=4):
    prompts = [
        "def fibonacci(n):",
        "def sort_array(arr):",
        "class Stack:",
        "def binary_search(arr, target):",
    ]
    model.eval()
    score = 0.0
    for prompt in prompts[:n_samples]:
        ids = torch.tensor([tok.encode_prompt(prompt)]).to(device)
        out = model.generate(ids, max_new_tokens=40, n_loops=4, temperature=0.7)
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
    parser.add_argument("--resume", action="store_true")
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

    # ── Tokenizer ──
    tok = BPETok(8192, "tokenizer.json")
    if not os.path.exists("tokenizer.json") and os.path.exists("bpe_tokenizer_8k.json"):
        os.rename("bpe_tokenizer_8k.json", "tokenizer.json")
        tok = BPETok(8192, "tokenizer.json")
    if not os.path.exists("tokenizer.json"):
        print("Training tokenizer...")
        sample = _load_or_download("wiki", ("Salesforce/wikitext", "wikitext-2-v1"), False, 0, None)[:3000]
        sample += _load_local_leetcode()[:500]
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
    cfg = TinyConfig()
    model = TinyModel(cfg).to(device)
    print(f"Params: {model.num_parameters():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = WSDScheduler(opt, warmup_steps=args.warmup, peak_lr=args.lr)
    crit = nn.CrossEntropyLoss()

    # Validation loader
    val_ds = FlatDataset(flat_val, args.seq_len)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Stage boundaries
    stage1_end = int(args.max_steps * args.stage1_pct)
    stage2_end = stage1_end + int(args.max_steps * args.stage2_pct)

    # ── Resume ──
    step, best_loss = 0, float("inf")
    if args.resume and os.path.exists("checkpoints/latest.pt"):
        ckpt = torch.load("checkpoints/latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer"])
        sched._step = ckpt.get("step", 0)
        step = ckpt.get("step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        sched._mode = ckpt.get("sched_mode", "warmup")
        print(f"Resumed at step {step}")

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
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0, drop_last=True)
    train_iter = iter(train_dl)
    current_stage = 0  # 0=stage1, 1=stage2, 2=stage3

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
        logits, aux_loss = model(x, n_loops=4)
        nll = crit(logits.view(-1, cfg.vocab_size), y.view(-1))
        loss = nll + args.aux_scale * aux_loss
        loss = loss / args.grad_acc
        loss.backward()

        if (step + 1) % args.grad_acc == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        nll_val = nll.item()

        if args.cooldown > 0:
            time.sleep(args.cooldown)

        if step % args.log_every == 0:
            lr = opt.param_groups[0]["lr"]
            stage_names = ["S1", "S2", "S3"]
            print(f"S{step} [{stage_names[current_stage]}] loss={nll_val:.4f} lr={lr:.2e}")

        if step > 0 and step % args.save_every == 0:
            v = evaluate(model, val_dl, crit, device, args.eval_steps)
            code_acc = eval_code_syntax(model, tok, device)
            if v < best_loss:
                best_loss = v
                torch.save({"model_state": model.state_dict()}, "checkpoints/best.pt")
                print(f"  ★ Best val_loss={v:.4f} code_acc={code_acc:.2f}")
                stall = 0
            else:
                stall += 1
                print(f"  val_loss={v:.4f} code_acc={code_acc:.2f} best={best_loss:.4f} stall={stall}/{args.patience}")
            torch.save({
                "model_state": model.state_dict(),
                "optimizer": opt.state_dict(),
                "sched_mode": sched._mode,
                "step": step, "best_loss": best_loss,
            }, "checkpoints/latest.pt")
            model.train()
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
