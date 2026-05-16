
## 2026-05-10 — Genes restructured to genes/ directory

### What Happened
All gene skills were living under `~/.opencode/skills/` mixed with non-gene skills, making them hard to find and manage.

### Root Cause
Genes were created in the skills directory by default, with no dedicated genes/ directory.

### Fix Applied
Created `~/.opencode/genes/` and moved all 17 gene and evolver directories there. Updated AGENTS.md and test_gene_status.py to scan genes/ instead of skills/.

### Pattern Learned
**Pattern name**: Genes deserve their own directory, separate from skills
**When to apply**: When creating new gene-type evomap skills
**How to apply**: Create under `~/.opencode/genes/evomap-gene-<name>/SKILL.md`

### Core Safe?
- [x] No core architecture affected
- [x] All tests pass (47/47)

---
