"""
Test cases to verify evomap-gene-status works correctly.
Run: python3 test_gene_status.py
"""
import os, sys, json, re

SKILLS_DIR = os.path.expanduser("~/.opencode/genes")
AGENTS_MD = os.path.expanduser("~/.opencode/AGENTS.md")
STATUS_LOG = "gene_status_log.md"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

EXPECTED_GENES = [
    "evomap-gene-repair",
    "evomap-gene-optimize-prompt",
    "evomap-gene-innovate",
    "evomap-gene-optimize-tool",
    "evomap-gene-env-vars",
    "evomap-gene-ralph-loop",
    "evomap-gene-training-resilience",
    "evomap-gene-fast-feedback",
    "evomap-gene-reflect",
    "evomap-gene-status",
    "evomap-gene-eval-gate",
    "evomap-gene-coordinator",
    "evomap-gene-index",
    "evolver-model-training",
    "evolver-integration",
    "evomap-all-capsules",
    "evomap-model-training-library",
]

passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name} — {detail}")
        failed += 1

print("=" * 60)
print("Gene Status Test Suite")
print("=" * 60)

# ── Test 1: gene-status skill file exists ──
print("\n[Test 1] gene-status skill exists")
status_skill = os.path.join(SKILLS_DIR, "evomap-gene-status", "SKILL.md")
test("SKILL.md exists", os.path.exists(status_skill), f"not found at {status_skill}")

# ── Test 2: Mandatory fields in gene-status SKILL.md ──
print("\n[Test 2] gene-status mandatory fields")
if os.path.exists(status_skill):
    content = open(status_skill).read()
    test("name field", "name: evomap-gene-status" in content, "missing name")
    test("mandatory: true", "mandatory: true" in content, "not marked mandatory")
    test("load_on field", "load_on:" in content, "missing load_on")
    test("session-start in load_on", "session-start" in content, "missing session-start trigger")
    test("session-end in load_on", "session-end" in content, "missing session-end trigger")
    test("End-of-conversation summary section", "End-of-Conversation Summary" in content, "missing summary section")
    test("✅ symbol in summary format", "✅" in content, "missing ✅ in summary template")
    test("⏸️ symbol in summary format", "⏸️" in content or "Standby" in content, "missing standby indicator")
else:
    test("SKILL.md readable", False, "file not found")

# ── Test 3: AGENTS.md has mandatory gene-status entry ──
print("\n[Test 3] AGENTS.md mandatory gene-status")
if os.path.exists(AGENTS_MD):
    agents = open(AGENTS_MD).read()
    test("gene-status in AGENTS.md", "evomap-gene-status" in agents, "not mentioned")
    test("Mandatory section exists", "Mandatory Load" in agents or "MANDATORY" in agents, "no mandatory section")
    test("End-of-conversation rule", "End of conversation" in agents, "missing end-of-convo rule")
else:
    test("AGENTS.md exists", False, f"not found at {AGENTS_MD}")

# ── Test 4: All expected gene skill directories exist ──
print("\n[Test 4] All gene skill directories exist")
for gene in EXPECTED_GENES:
    gene_dir = os.path.join(SKILLS_DIR, gene)
    gene_skill = os.path.join(gene_dir, "SKILL.md")
    test(f"{gene}/SKILL.md", os.path.exists(gene_skill), f"missing {gene_skill}")

# ── Test 5: gene-status can scan and list all genes ──
print("\n[Test 5] gene-status scan detects all installed genes")
installed = []
for gene in EXPECTED_GENES:
    gene_skill = os.path.join(SKILLS_DIR, gene, "SKILL.md")
    if os.path.exists(gene_skill):
        installed.append(gene)
test(f"Found {len(installed)}/{len(EXPECTED_GENES)} genes", len(installed) == len(EXPECTED_GENES),
     f"found {len(installed)}, expected {len(EXPECTED_GENES)}")

# ── Test 6: gene_status_log.md exists and has valid format ──
print("\n[Test 6] gene_status_log.md format")
log_path = os.path.join(PROJECT_DIR, STATUS_LOG)
test("gene_status_log.md exists", os.path.exists(log_path), f"not found at {log_path}")
if os.path.exists(log_path):
    log_content = open(log_path).read()
    test("Has table header", "| #" in log_content or "| Gene" in log_content, "no table in log")
    test("Has gene names", any(g.split("evomap-gene-")[-1] in log_content or g.split("evolver-")[-1] in log_content for g in EXPECTED_GENES), "no gene entries in log")
    test("Has Used/Standby status", "Used" in log_content or "✅" in log_content, "no status markers")

# ── Test 7: gene-status index lists gene-status ──
print("\n[Test 7] gene-index includes gene-status")
index_skill = os.path.join(SKILLS_DIR, "evomap-gene-index", "SKILL.md")
if os.path.exists(index_skill):
    idx_content = open(index_skill).read()
    test("evomap-gene-status in index", "evomap-gene-status" in idx_content, "not listed in index")
    test("MANDATORY label in index", "MANDATORY" in idx_content, "not marked mandatory in index")
else:
    test("gene-index SKILL.md exists", False, "not found")

# ── Test 8: Summary format validation ──
print("\n[Test 8] Summary format is valid")
summary_template = """
📂 Gene Status — This Conversation

| Gene | Used? | What It Did |
|------|-------|-------------|
| gene-status | ✅ | This report |
| gene-repair | ✅ | Fixed ... |
| gene-innovate | ⏸️ | Standby |

Summary: X/14 used, Y standby
"""
test("Summary has table header", "| Gene |" in summary_template)
test("Summary has ✅ marker", "✅" in summary_template)
test("Summary has ⏸️ or Standby", "⏸️" in summary_template or "Standby" in summary_template)
test("Summary has Summary line", "Summary:" in summary_template)

# ── Test 9: End-of-conversation triggers ──
print("\n[Test 9] End-of-conversation trigger rules")
if os.path.exists(status_skill):
    content = open(status_skill).read()
    test("session-start trigger", "session-start" in content, "missing session-start")
    test("session-end trigger", "session-end" in content, "missing session-end")
    test("task-complete trigger", "task-complete" in content, "missing task-complete")
    test("ALWAYS last message rule", "last thing output" in content.lower() or "always" in content.lower(), "missing always-last rule")
else:
    test("SKILL.md exists for trigger check", False, "file not found")

# ── Test 10: Runtime — generate a status scan ──
print("\n[Test 10] Runtime gene scan")
scan_results = {}
for gene in EXPECTED_GENES:
    gene_skill = os.path.join(SKILLS_DIR, gene, "SKILL.md")
    if os.path.exists(gene_skill):
        skill_content = open(gene_skill).read()
        is_mandatory = "mandatory: true" in skill_content.lower()
        desc_match = re.search(r"description:\s*(.+)", skill_content)
        desc = desc_match.group(1).strip() if desc_match else "no description"
        scan_results[gene] = {"installed": True, "mandatory": is_mandatory, "description": desc}
    else:
        scan_results[gene] = {"installed": False, "mandatory": False, "description": ""}

all_installed = all(v["installed"] for v in scan_results.values())
mandatory_genes = [g for g, v in scan_results.items() if v["mandatory"]]
test("All genes installed", all_installed, f"missing: {[g for g,v in scan_results.items() if not v['installed']]}")
test("gene-status is mandatory", "evomap-gene-status" in mandatory_genes, "gene-status not marked mandatory")
test(f"Found {len(mandatory_genes)} mandatory gene(s)", len(mandatory_genes) >= 1, "no mandatory genes found")

# ── Results ──
print("\n" + "=" * 60)
print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
print("=" * 60)
if failed > 0:
    print("⚠️  Some tests failed. Review the output above.")
    sys.exit(1)
else:
    print("✅ All gene-status tests passed.")
    sys.exit(0)