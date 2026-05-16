# Training Progress

**Model: Theos v2 — 63M params** (up from 36M)
**Started:** 2026-05-16

## Config Diff

| Param | v1 (old) | v2 (new) |
|-------|----------|----------|
| dim | 1280 | 1280 |
| n_heads | 20 | 20 |
| n_blocks | 1 | 2 |
| params | 36.7M | 63.0M |
| max_steps | 350k | 500k |

## Current Status

| Metric | Value |
|--------|-------|
| Step | 2,000 / 500,000 (0.4%) |
| Best val_loss | 7.2443 |
| Train loss | 3.618 |
| LR | 3.00e-04 (stable) |
| Stage | 1 — General + Broad Code |
| Code eval | 0 pass (too early) |

## Loss Trend

| Step | Train | Val |
|------|-------|-----|
| 0 | 9.03 | - |
| 200 | 4.78 | - |
| 600 | 4.21 | - |
| 1000 | 3.91 | - |
| 1400 | 3.62 | - |
| 2000 | 3.62 | 7.24 |

## Next Milestones

- S5,000 — next eval checkpoint
- S210,000 — Stage 1 → Stage 2 (60% mark)
- S297,500 — Stage 2 → Stage 3 (85% mark)
- S500,000 — completion

## v1 Results (for comparison)

- Final step: 350,000
- Best val_loss: 5.2901
- Code eval: 0/8 passed
- Plateaued at val_loss ~5.30 since S250k
