# Theos

Small language model with Recurrent-Depth Transformer + Linear Attention.

## Multi-Stage Training Strategy

Inspired by SmolLM2's data-centric approach. Three stages with WSD scheduler.

### Stage 1 — General + Broad Code (60% of steps)
- Warmup (200 steps) + stable LR at 3e-4
- Mixture: codeparrot-clean (45%), FineWeb (25%), CodeAlpaca (12%), WikiText (8%), LeetCode (10%)
- Purpose: Build language understanding and basic coding patterns

### Stage 2 — Reasoning + Specialized Code (25% of steps)
- Stable LR at 3e-4
- Mixture: codeparrot-clean (30%), LeetCode (15%), Dolphin-Coder (15%), CodeFeedback (15%), SmolTalk (15%), synthetic (10%)
- Purpose: Strengthen reasoning with CoT traces and diverse instructions

### Stage 3 — High Quality + Annealing (15% of steps)
- Linear LR decay to 0 over this stage
- Mixture: Self-OSS-Instruct (30%), LeetCode (20%), synthetic (15%), SmolTalk (15%), codeparrot-clean (20%)
- Purpose: Polish on highest-quality data — biggest capability jump

## Data Sources

| Dataset | Source | Size | Type |
|---------|--------|------|------|
| codeparrot-clean | Hugging Face | 500k files (4.9 GB) | Raw multi-language code |
| Self-OSS-Instruct | bigcode | 50.7k | Python code instructions |
| CodeFeedback | m-a-p | 50k | Code instructions |
| Dolphin-Coder | cognitivecomputations | 50k | Code instructions |
| SmolTalk | HuggingFaceTB | 100k | SmolLM2 instruction dataset |
| LeetCode (various) | Local | ~10k | Code problems with CoT |
| FineWeb | HuggingFaceFW | 100k | General web text |
| CodeAlpaca | sahil2801 | 20k | Code instructions |
| WikiText | Salesforce | ~36k | General text |
| OpenAssistant | timdettmers | 15k | Assistant conversations |
| Synthetic | Generated | ~240 | Synthetic reasoning |

## Architecture

- Recurrent-depth transformer with Kim Linear Attention
- 4 loop iterations, ACT halting
- MoE FFN (8 experts, top-2 routing)
- BPE tokenizer (8k vocab)
- 6.6M parameters

## Training

```bash
# Fresh train (default 50k steps, ~108M tokens)
python3 train.py

# Quick test (300 steps)
python3 train.py --quick

# Micro sanity check
python3 train.py --micro
```
