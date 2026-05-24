# Gene Status Log

## 2026-05-10T10:30 — Theos Training Session

**Project**: theos (TinyModel recurrent-depth transformer training)
**Task**: Train model, fix generate() bug, apply all gene optimizations, retrain

| # | Gene | Status | Applied This Run |
|---|------|--------|-------------------|
| 1 | gene-repair | Active+Used | Fixed `_load_local_leetcode` NameError bug |
| 2 | gene-innovate | Active+Used | Root-cause diagnosed generate() T=1 no-context bug |
| 3 | gene-optimize-prompt | Active+Standby | Available for data/prompt optimization |
| 4 | gene-optimize-tool | Active+Used | Batched tool calls, reduced sequential ops |
| 5 | gene-env-vars | Active+Used | Found `ENABLE_CUDA_GRAPH=1` env var |
| 6 | gene-ralph-loop | Active+Standby | PRD framework ready for structured iteration |
| 7 | gene-training-resilience | Active+Used | Created `run.sh` auto-restart, checkpoint every 200 steps, auto-resume logic |
| 8 | gene-fast-feedback | Active+Used | Added smoke test gate to `train.py`, caught generate() degeneration, created `eval_real.py` |
| 9 | gene-index | Active+Used | Routed signals to correct genes throughout session |
| 10 | evolver-model-training | Active+Used | Applied 7 patches: GradScaler+bfloat16 fix, vectorized repetition penalty, precomputed causal mask buffer, FlatDataset torch pre-convert, dropout 0.2→0.1, persistent_workers, _load_local_leetcode fix |
| 11 | evolver-integration | Active+Used | Guided full diagnostic cycle, coordinated gene activation |
| 12 | evomap-all-capsules | Active+Standby | Loaded but no specific capsule selected |
| 13 | evomap-model-training-library | Active+Standby | Loaded but no library pattern applied |
| 14 | gene-status | Active+Used | Created this report, mandatory gene tracker |

**Summary**: 10/14 genes Active+Used, 4/14 Active+Standby, 0 Not Installed

### Key Changes This Session
- **tiny_model.py**: Fixed generate() (full-sequence processing), vectorized repetition penalty, precomputed causal mask buffer, dropout 0.2→0.1
- **train.py**: Fixed `_load_local_leetcode` bug, removed GradScaler for bfloat16, added smoke test, auto-resume from checkpoint, persistent_workers, FlatDataset torch pre-convert
- **run.sh**: Auto-restart wrapper for crash recovery
- **eval_real.py**: Real code eval (generate + compile + execute + test)
- **Gene skills**: Created gene-training-resilience, gene-fast-feedback, gene-status

---

## 2026-05-17T02:06 — Theos Training Session (Current)

**Project**: theos (TinyModel recurrent-depth transformer training)  
**Task**: Monitor training progress, quick eval, gene status session audit

| # | Gene | Status | Applied This Run |
|---|------|--------|-------------------|
| 1 | gene-repair | Active+Standby | No code fixes needed this session |
| 2 | gene-innovate | Active+Standby | No new features or root-cause analysis |
| 3 | gene-optimize-prompt | Active+Standby | Pending data/prompt optimization |
| 4 | gene-optimize-tool | Active+Used | Batched scan + log reads efficiently |
| 5 | gene-env-vars | Active+Standby | No .env or env var handling needed |
| 6 | gene-ralph-loop | Active+Standby | No PRD-driven iteration this session |
| 7 | gene-training-resilience | Active+Used | Monitored checkpointing (auto-resume active), latest.pt at S~3300 |
| 8 | gene-fast-feedback | Active+Used | Quick eval of training loss (loss oscillating ~3.5, not dropping) |
| 9 | gene-index | Active+Used | Routed signals to correct genes (eval → fast-feedback) |
| 10 | evolver-model-training | Active+Standby | Monitored but did not patch training code this session |
| 11 | evolver-integration | Active+Standby | No full diagnostic cycle initiated |
| 12 | gene-coordinator | Active+Used | Autonomous monitor active, checked quality, reported status |
| 13 | gene-eval-gate | Active+Used | Checkpoint eval: lossVolatile, steps=3200, verdict=needsOptimization |
| 14 | gene-status | Active+Used | This report — mandatory gene tracker invoked |
| 15 | evomap-all-capsules | Active+Standby | No marketplace capsule applied |
| 16 | evomap-model-training-library | Active+Standby | No library pattern applied |

**Summary**: 6/16 genes Active+Used, 10/16 Active+Standby, 0 Not Installed

### Key Observations This Session
- **Training Health**: Step ~3300 / 500000, loss oscillating 2.8–4.9 (avg ~3.5). Not dropping steadily.
- **Checkpointing**: `latest.pt` (~761MB) saved May 17 11:54, auto-resume from S2000 confirmed.
- **GENE-EVAL-GATE**: Loss not improving past 30K steps protocol. Verdict: **needsOptimization** — LR or data mixing issue likely.
- **EvoMap Gene-Coordinator**: Monitored, no code changes applied (training still running).

---

## 2026-05-17T21:27:14.UTC

## Gene Status — This Conversation

| Gene | Used? | What It Did |
|------|-------|-------------|
| evomap-gene-status | ✅ | Loaded and generating this report |
| evomap-gene-index | ✅ | Loaded for signal routing (mandatory) |
| evomap-gene-training-resilience | ✅ | Started resilient training loop via run.sh |

Summary: 3/14 used, 11 standby

