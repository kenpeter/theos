# Theos

Small language model — Recurrent-Depth Transformer with Linear Attention.
Trained on 1.86B tokens across 3 stages. phi-1 philosophy: small model, max data.

## Architecture

```
Input → [Prelude ×4: StandardAttention + SwiGLU FFN]
       → [RecurrentBlock ×1 (looped 4x): GatedLinearAttention + MoE-FFN (8 experts, top-2)]
       → [Coda ×4: StandardAttention + SwiGLU FFN]
       → Output
```

| Config | Value |
|--------|-------|
| Params | ~35M |
| Dim | 512 |
| Heads | 8 (head_dim=64) |
| Seq len | 1024 |
| Vocab | 8192 BPE tokens |
| Recurrent loops | 4 (weight-tied) |
| ACT halting | threshold=0.9 |
| MoE | 8 experts, top-2 |
| LoRA per-loop | rank=16 |

## Multi-Stage Training (WSD Scheduler)

Three stages inspired by SmolLM2 data curriculum:

### Stage 1 — General + Broad Code (60% of steps, ~30k)
- Warmup 200 steps → stable LR 3e-4
- Heavy on codeparrot-clean + FineWeb
- Purpose: language modeling + broad code patterns

### Stage 2 — Reasoning + Specialized Code (25% of steps, ~12.5k)
- Stable LR 3e-4
- Shift toward LeetCode, Dolphin-Coder, CodeFeedback
- Purpose: CoT reasoning, instruction following, algorithmic thinking

### Stage 3 — High Quality + Annealing (15% of steps, ~7.5k)
- Linear LR decay 3e-4 → 0
- Highest quality data: Self-OSS-Instruct, LeetCode, synthetic
- Purpose: polish — biggest capability jump per step

## Data Sources

| Dataset | Count | Type |
|---------|-------|------|
| codeparrot_clean | 500k | Multi-language code |
| Self-OSS-Instruct | 50,661 | Python code instruct |
| CodeFeedback | 50k | Code instructions |
| Dolphin-Coder | 50k | Code instructions |
| SmolTalk | 50k (trimmed) | Instruction data |
| Magicoder | 50k (sampled) | Evol-instruct code |
| FineWeb | 50k (sampled) | General web text |
| LeetCode (9 datasets) | ~66k | Code problems + CoT |
| CodeAlpaca-20k | ~20k | Code instructions |
| WikiText-2 | ~16k | General text |
| OpenAssistant Guanaco | ~15k | Assistant convos |
| Roleplay | ~large | Dialogue text |
| Harvested LeetCode | ~3.5k | Code + reasoning |
| Synthetic | ~240 | Hand-crafted reasoning |

**Total: 1.12M training texts → 1.86B tokens**

## Training

```bash
# Fresh train (50k steps, 3 stages)
python3 train.py --batch_size 2

# Resume from latest checkpoint
python3 train.py --batch_size 2 --resume

# Quick test (300 steps)
python3 train.py --quick

# Micro sanity check
python3 train.py --micro

# Tokenize data only (no training)
python3 train.py --data_only
```
