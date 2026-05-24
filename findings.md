# Training Failure Analysis

## Executive Summary

The model fails to produce working code because the training pipeline has **6 critical bugs** that prevent any meaningful code generation learning. The 38% compile rate on `best.pt` is barely above random, and the regression to 0% on `latest.pt` is expected behavior — the model is overfitting to noise, not learning to write code.

---

## Critical Issues

### C1: seq_len=256 is far too short for code

**`train.py:434`** (passed as `--seq_len 256`)

Each training sample is only 256 tokens. But a single `def fibonacci(n):` with the `<|user|>...<|assistant|>...` prompt format takes ~105 tokens. After prompt overhead, only ~150 tokens remain for the function body — the model **never sees a complete multi-line function** during training. Meanwhile, `TinyConfig.max_seq_len=1024`, so the model can generate longer sequences at inference but has never been trained to do so coherently.

### C2: No document boundaries — FlatDataset cross-contamination

**`train.py:73-87`**

```python
class FlatDataset(Dataset):
    def __getitem__(self, i):
        x = self.tokens[i : i + self.seq_len]
        y = self.tokens[i + 1 : i + self.seq_len + 1]
```

All texts are concatenated into one flat 1D array. A 256-token chunk can start mid-document and end mid-document. The model is trained to predict tokens **across document boundaries** — it learns to continue from `<|end|>` into the start of the next unrelated sample. This is training on garbage for significant portions of the data.

### C3: No loss masking

**`train.py:591`**

```python
crit = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
```

There is no `ignore_index`. ALL tokens contribute equally to the loss:
- Special tokens (`<pad>`, `<eos>`, `<unk>`, `<bos>`)
- Formatting tokens (`<|user|>`, `<|assistant|>`, `<|end|>`)
- Prompt/problem description tokens (which should not be predicted)
- Cross-document boundary tokens (impossible to predict correctly)

The model wastes 50%+ of its capacity on special tokens, formatting, and garbage boundaries.

### C4: AMP is completely disabled

**`train.py:592,678`**

```python
scaler = None  # bfloat16 doesn't need GradScaler
...
with torch.amp.autocast(device_type=device.type, enabled=scaler is not None, dtype=torch.bfloat16):
```

Since `scaler is None`, `enabled=False`. The `--amp` flag does nothing — training runs in **full float32**. The comment is incorrect: bfloat16 autocast works without GradScaler, but `enabled=True` is still required.

### C5: Architecture changed between runs (checkpoint incompatibility)

| Run | Parameters | Date | Val Loss |
|-----|-----------|------|----------|
| `checkpoints_old/best.pt` | 8,759,606 | May 6 | Unknown |
| `checkpoints_old/v1_36m/best.pt` | 48,317,282 | May 13 | 5.29 (step 290K) |
| `training_run.log` (May 13) | 36,750,178 | May 13 | — |
| `training.log` (May 17) | 63,015,876 | May 17 | — |
| Current `best.pt` / `latest.pt` | **74,582,980** | May 18-24 | 5.66 (step 35K) |

The model architecture changed **4 times**. The best overall checkpoint (val_loss=5.29 at step 290K from the 36M-param run) was overwritten. Checkpoints between architectures are incompatible — each architecture change means starting from scratch.

### C6: batch_size=1 with noisy mixed data

**`train.py:433`** (`--batch_size 1`)

Extremely noisy gradients with no batch averaging. Combined with 30% non-code data (243K roleplay fiction texts, wiki, web forums, chat), the model overfits to specific patterns and memorizes noise rather than learning generalizable code generation.

---

## High-Severity Issues

### H1: 30% of training data is non-code

**`train.py:178-358,500-549`**

| Dataset | Texts | Type |
|---------|-------|------|
| Wiki (wikitext-2) | 16,180 | General text |
| FineWeb (sample-10BT) | 50,000 | Web forum posts |
| OpenAssistant (guanaco) | 15,000 | Chat dialogue |
| **Bluemoon Roleplay** | **243,193** | **Fiction/story — 22% of total data** |
| Code datasets | ~770,000 | Code |

~324K out of ~1.1M texts (~30%) are non-code. The model learns to predict roleplay fiction and forum posts. When prompted with a code request, it may generate roleplay or chat text instead.

### H2: Val_loss does not correlate with code quality

| Step | Val Loss | Compile Rate | Tests Passing |
|------|----------|-------------|---------------|
| 220K | 5.382 | 88% | 0/8 |
| 240K | 5.322 | 50% | 0/8 |
| 290K | 5.290 | 25% | 0/8 |
| 35K (current best) | 5.663 | 37% | 0/8 |
| 39.4K (latest) | 5.663+ | 0% | 0/8 |

Val loss decreases while code quality fluctuates wildly. The loss function (next-token prediction on mixed-domain 256-token chunks) has **no direct relationship** with generating compilable Python code. Using val_loss to select the "best" checkpoint is meaningless.

### H3: WSDScheduler uses hardcoded 500K steps

**`train.py:100,107-108`**

```python
self._warmup_end = int(total_steps * 0.60)
self._stable_end = int(total_steps * 0.85)
```

`total_steps` defaults to 500,000 and is hardcoded, never passed from the caller. If `--max_steps` is changed, the scheduler ignores it. Steps 0-300K are labeled "warmup" (but LR is at peak for 299,800 of those), steps 300-425K are "stable", then cosine decay to 500K.

### H4: Adaptive LR decay never triggers during warmup

**`train.py:734`**

```python
if lr_stall >= args.lr_patience and sched._mode == "stable":
```

For the current run at step 35-39K, the scheduler is in "warmup" mode (step ≤ 300K). Adaptive LR decay on plateau **never fires** during this period, even if the loss is completely flat for thousands of steps.

### H5: Eval code extraction captures incomplete functions

**`eval_real.py:87`**

```python
match = re.search(rf'(def\s+{fn_name}\s*\([^)]*\)\s*:[^\n]*(?:\n[ \t]+[^\n]*)*)', p_text, re.DOTALL)
```

This regex captures the `def` line plus any lines that start with whitespace. It **does not capture** blank lines within the function, lines at column 0, multi-line strings, or decorators. The capture stops at the first non-indented line — typically only 1-3 lines of incomplete function body.

### H6: --cooldown 1.0 wastes 5.8 days

**`train.py:700-701`** (passed as `--cooldown 1.0`)

```python
if args.cooldown > 0:
    time.sleep(args.cooldown)
```

1 second sleep per training step. Over 500K steps, that's ~5.8 days of sleeping. For the 4,400 steps from 35K to 39.4K, that's 73 minutes wasted with zero benefit.

### H7: Tokenizer cache doesn't invalidate on data change

**`train.py:49-50,557`**

```python
if cache_path and os.path.exists(cache_path):
    arr = np.load(cache_path)
    return arr
```

If datasets are added, removed, or changed, the cached `.npy` files are silently reused. Only manual deletion forces re-tokenization.

### H8: Stage transitions are cosmetic

**`train.py:650-668`**

The three-stage training (general → code+reasoning → high quality) does **nothing** — stages don't swap datasets, the scheduler handles decay independently, and the stage logic just reshuffles the dataloader iterator.

---

## Medium-Severity Issues

### M1: Backwards variable name

**`train.py:761`**

```python
_compile_ok = _eg.stdout.count("syntax error") + _eg.stdout.count("no valid def")
```

`_compile_ok` actually counts **failures** (syntax errors and missing definitions). The formula is correct but the name is dangerously misleading.

### M2: Label smoothing hurts code prediction

**`train.py:591`** (`--label_smooth 0.1`)

Label smoothing blurs the target distribution, making the model less confident about exact next-token predictions. For code generation, where exact tokens are required, this is actively harmful.

### M3: High weight decay

**`train.py:589`** (`--weight_decay 0.1`)

Very aggressive regularization for a 74.6M parameter model. May over-regularize and prevent the model from learning detailed code patterns.

### M4: Gradient accumulation broken for grad_acc > 1

**`train.py:682-697`**

```python
loss = loss / args.grad_acc
...
if (step + 1) % args.grad_acc == 0:
    opt.step()
```

The loss is divided every step, but optimizer steps only every `grad_acc` steps. For `grad_acc > 1`, the accumulated loss is incorrectly scaled since division happens per-step but accumulation doesn't sum.

### M5: Validation set uses last N examples (systematic bias)

**`train.py:514-549`**

Validation takes the last N examples from each dataset. If datasets are sorted by complexity or length, the validation set may contain disproportionately long or complex texts.

### M6: InfiniteSampler uses tiny random batches

**`train.py:633-642`**

```python
yield from (int(i) for i in torch.randint(0, self.n, (self.n // 1000 + 1,)))
```

For `n ≈ 7M samples`, this yields batches of ~7,000 random indices at a time. Not uniform over long training runs.

### M7: Validation split differs between train.py and train_fixed.py

**`train.py:520`**: `split = int(len(leet) * 4 / 5)` (1/5 for val)
**`train_fixed.py:492`**: `split = int(len(leet) * 19 / 20)` (1/20 for val)

Different training scripts use different validation splits on the same data, meaning checkpoint comparisons across runs are invalid.

---

## The Core Regression: Why 38% → 0%

Between step 35K (best.pt) and step 39.4K (latest.pt), 4,400 additional training steps caused:

1. **Catastrophic drift**: With batch_size=1 and mixed data, the model rapidly overfits to specific token patterns. The additional steps pushed the model away from the accidental patterns that produced compilable code at step 35K.

2. **Generation quality collapsed**: latest.pt output shows nonsense function names (`is_str_len`, `fibons2i`), garbled number sequences (`1234567139710245689139720456800`), and broken indentation — all classic signs of overfitting.

3. **No checkpoint improvement**: Val_loss stayed at 5.66 (no improvement), so `best.pt` was never updated. The model got worse at code generation without any corresponding signal in the loss.

---

## Recommended Fixes (Priority Order)

### P0: Fix the training paradigm

- Replace `FlatDataset` with **document-aware chunking**: each sample = one complete `<|user|>...<|assistant|>...<|end|>` example
- Add **loss masking**: `ignore_index` on prompt/special tokens, only compute loss on assistant response
- Increase **seq_len** to at least 512, ideally 1024 (matching `max_seq_len`)
- Remove or drastically reduce non-code data (roleplay, wiki, web, chat)
- Increase effective batch size: `batch_size >= 8` with `grad_acc >= 4`

### P1: Fix AMP

```python
scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
```

### P2: Fix WSDScheduler

Pass `total_steps` from `--max_steps` arg. Fix mode logic so adaptive LR decay works throughout training.

### P3: Fix evaluation

- Increase `max_new_tokens` to 300-500
- Improve regex to capture full function body (including blank lines)
- Remove hacky fixup code

### P4: Remove `--cooldown`

No benefit, destroys throughput.

### P5: Recover best checkpoint from `checkpoints_old/v1_36m/`

Val_loss=5.29 at step 290K (vs current 5.66 at step 35K) — significantly better even if from a different architecture.
