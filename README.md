# Theos

## Forward Pass

```
Input BPE Tokens (max_len=1024)
    │
    ▼
  Embed  8192 → 1280  (tied with Head)
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│                    LoopedBlock (weight-tied)                  │
│                                                              │
│  for t in 0..n_loops:                                        │
│    │                                                         │
│    ├─ loop_index_embedding(t) ─→ h                           │
│    ├─ RMSNorm(h + e)                                         │
│    │      │                                                  │
│    │      ├─► Multi-Head Attention (20 heads × dim=64)       │
│    │      │     · RoPE (θ=10000)                             │
│    │      │     · Causal mask                                │
│    │      │     · SDPA flash attention                       │
│    │      │                                                  │
│    │      ├─► SwiGLU FFN (dim×4 hidden)                      │
│    │      │     gate(x) → silu → * up(x) → down              │
│    │      │                                                  │
│    │      └─► LoRA Adapter (rank=16)                         │
│    │            down(1280→16) × scale(t) × B(16→1280)        │
│    │                                                         │
│    ├─ residual + dropout(0.1)                                │
│    │                                                         │
│    ├─ ACT Halting                                            │
│    │   p ← σ(halt_proj(h))      halted when Σp ≥ 0.9         │
│    │   output ← Σ p_t · h_t                                  │
│    │                                                         │
│    └─ LTI Injection                                          │
│        A ← exp(-exp(log_dt + log_A))                         │
│        h ← A·h + B·e + trans_out                             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
  RMSNorm
    │
    ▼
  Head  1280 → 8192  (tied with Embed)
    │
    ▼
  Logits → Top-k(40) · Temp(0.8) · RepPenalty(1.2) → token
```

## Training Schedule (WSD)

```
  LR
   │
3e-4 ─────────────────────●──────────●
   │                    ╱              ╲
   │                  ╱                  ╲
   │                ╱                      ╲
   │              ╱                          ╲
 0 ─┼─────────●──┼───────────────────────────●───► steps
   0         200       210k        297.5k    350k
   │         │         │           │          │
   │ Warmup  │  Stable │  Stable   │  Anneal  │
   │         │         │           │          │
   │ general │ general │ reasoning │ high-q   │
   │ + code  │ + code  │ + code    │ only     │
```

```
plateau detected (3 evals no Δ) → LR × 0.5
```

## Data Pipeline

```
Raw texts
    │
    ▼
 BPETokenizer.train(vocab=8192)  →  encode_prompt()
    │
    ▼
 flat int32 array
    │
    ▼
 FlatDataset (sliding window, seq_len=1024)
    │
    ▼
 InfiniteSampler  →  DataLoader  →  training loop
```

## Specs

| Param | Value | | Param | Value |
|-------|-------|--|-------|-------|
| Total params | 36.7M | | LoRA rank | 16 |
| Embed dim | 1280 | | Dropout | 0.1 |
| Heads | 20 (d=64) | | Max loops | 6 |
| Vocab size | 8192 (BPE) | | ACT halt threshold | 0.9 |
| Max seq len | 1024 | | Weight tying | embed ↔ head |
| Training loops | 4 | | Rep penalty | 1.2 |

## Generation

```
seed_prompt → encode → for each new token:
    │                       │
    ▼                       ├─ full seq re-encoded
  full seq ───────────────► ├─ RepPenalty(1.2)
                            ├─ Top-k(40)
                            └─ Temp(0.8) → sample → append
```
