"""Harvest all high-quality training data from local disk datasets."""
import pickle, json, os, glob
import pyarrow.parquet as pq
from pathlib import Path

CACHE = Path("dataset_cache")
CACHE.mkdir(exist_ok=True)

SEEN_SLUGS = set()
ALL_TEXTS = []

def add(slug, text):
    slug = slug.strip().lower().replace(" ", "-") if slug else ""
    if slug and slug in SEEN_SLUGS:
        return False
    if slug:
        SEEN_SLUGS.add(slug)
    ALL_TEXTS.append(text)
    return True

base = "/home/kenpeter/work/data"

# ── 1. mesolitica_LeetCodeQwQ — has QwQ CoT reasoning ──
print("=== mesolitica_LeetCodeQwQ ===")
try:
    df = pq.read_table(f"{base}/mesolitica_LeetCodeQwQ/train.parquet").to_pandas()
    for _, r in df.iterrows():
        slug = r.get("slug", "")
        content = r.get("content", "")
        solution = r.get("solution", "")
        qwq = r.get("qwq", "")
        text = f"<|user|>\n{content}\n<|assistant|>\n{solution}\n<|end|>"
        add(slug, text)
    print(f"  loaded {sum(1 for _ in range(len(df)))}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── 2. LimYeri_LeetCode — conversations format ──
print("=== LimYeri_LeetCode ===")
try:
    df = pq.read_table(f"{base}/LimYeri_LeetCode/train.parquet").to_pandas()
    for _, r in df.iterrows():
        title = r.get("title", "")
        convs = r.get("conversations", [])
        if isinstance(convs, list):
            user_text = ""
            asst_text = ""
            for c in convs:
                if isinstance(c, dict):
                    if c.get("from") == "user":
                        user_text = c.get("value", "")
                    elif c.get("from") == "assistant" or c.get("from") == "gpt":
                        asst_text = c.get("value", "")
            if user_text and asst_text:
                text = f"<|user|>\n{user_text}\n<|assistant|>\n{asst_text}\n<|end|>"
                add(title, text)
    print(f"  loaded subset from {len(df)} rows")
except Exception as e:
    print(f"  ERROR: {e}")

# ── 3. DenCT_LeetCode — with problem content and Python solutions ──
print("=== DenCT_LeetCode ===")
try:
    df = pq.read_table(f"{base}/DenCT_LeetCode/leetcode-java-python.parquet").to_pandas()
    for _, r in df.iterrows():
        slug = r.get("titleSlug", "")
        content = r.get("question_content", "")
        python_sol = r.get("Python3", "")
        if isinstance(python_sol, str) and python_sol.strip():
            text = f"<|user|>\n{content}\n<|assistant|>\n{python_sol}\n<|end|>"
            add(slug, text)
    print(f"  loaded subset from {len(df)} rows")
except Exception as e:
    print(f"  ERROR: {e}")

# ── 4. vovw_LeetCode — Python solutions ──
print("=== vovw_LeetCode ===")
try:
    df = pq.read_table(f"{base}/vovw_LeetCode/dataset.parquet").to_pandas()
    for _, r in df.iterrows():
        slug = r.get("slug", "")
        content = r.get("content", "")
        python_sol = r.get("python", "")
        if isinstance(python_sol, str) and python_sol.strip():
            text = f"<|user|>\n{content}\n<|assistant|>\n{python_sol}\n<|end|>"
            add(slug, text)
    print(f"  loaded subset from {len(df)} rows")
except Exception as e:
    print(f"  ERROR: {e}")

# ── 5. juyoungml_LeetCodeRosetta — Python code ──
print("=== juyoungml_LeetCodeRosetta ===")
try:
    df = pq.read_table(f"{base}/juyoungml_LeetCodeRosetta/train.parquet").to_pandas()
    for _, r in df.iterrows():
        title = r.get("title", "")
        content = r.get("content", "")
        python_code = r.get("python_code", "")
        if isinstance(python_code, str) and python_code.strip():
            text = f"<|user|>\n{content}\n<|assistant|>\n{python_code}\n<|end|>"
            add(title, text)
    print(f"  loaded subset from {len(df)} rows")
except Exception as e:
    print(f"  ERROR: {e}")

# ── 6. newfacade_LeetCodeDataset — jsonl with completions ──
print("=== newfacade_LeetCodeDataset ===")
try:
    with open(f"{base}/newfacade_LeetCodeDataset/leetcode_train.jsonl") as f:
        for line in f:
            d = json.loads(line)
            slug = d.get("task_id", "")
            content = d.get("problem_description", "")
            completion = d.get("completion", "")
            if content and completion:
                text = f"<|user|>\n{content}\n<|assistant|>\n{completion}\n<|end|>"
                add(slug, text)
    print(f"  loaded")
except Exception as e:
    print(f"  ERROR: {e}")

# ── 7. greengerong_LeetCode — multi-language solutions ──
print("=== greengerong_LeetCode ===")
try:
    with open(f"{base}/greengerong_LeetCode/leetcode-train.jsonl") as f:
        for line in f:
            d = json.loads(line)
            slug = d.get("slug", "")
            content = d.get("content", "")
            python_sol = d.get("python", "")
            if isinstance(python_sol, str) and python_sol.strip():
                text = f"<|user|>\n{content}\n<|assistant|>\n{python_sol}\n<|end|>"
                add(slug, text)
    print(f"  loaded")
except Exception as e:
    print(f"  ERROR: {e}")

# ── 8. high_quality_leetcode — already used but include more ──
print("=== high_quality_leetcode ===")
try:
    with open(f"{base}/high_quality_leetcode/train.jsonl") as f:
        for line in f:
            d = json.loads(line)
            slug = d.get("task_id", "")
            content = d.get("problem_description", "")
            completion = d.get("completion", "")
            cot = d.get("high_quality_cot", "")
            if cot:
                text = f"<|user|>\n{content}\n<|reasoning|>\n{cot}\n<|assistant|>\n{completion}\n<|end|>"
            elif content and completion:
                text = f"<|user|>\n{content}\n<|assistant|>\n{completion}\n<|end|>"
            else:
                continue
            add(slug, text)
    print(f"  loaded")
except Exception as e:
    print(f"  ERROR: {e}")

# ── 9. LeetCode_YT_CC_CoT_Summary — CoT summaries ──
print("=== LeetCode_YT_CC_CoT_Summary ===")
try:
    import csv
    csv.field_size_limit(2**31 - 1)
    with open(f"{base}/LeetCode_YT_CC_CoT_Summary/data.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slug = row.get("title_slug", "")
            qc = row.get("question_content", "")
            py = row.get("python", "")
            summary = row.get("Summary", "")
            if qc and py:
                if summary:
                    text = f"<|user|>\n{qc}\n<|reasoning|>\n{summary}\n<|assistant|>\n{py}\n<|end|>"
                else:
                    text = f"<|user|>\n{qc}\n<|assistant|>\n{py}\n<|end|>"
                add(slug, text)
    print(f"  loaded")
except Exception as e:
    print(f"  ERROR: {e}")

# Save
out_path = CACHE / "harvested_data.pkl"
pickle.dump(ALL_TEXTS, open(out_path, "wb"))
print(f"\nSaved {len(ALL_TEXTS)} texts ({sum(len(t.split()) for t in ALL_TEXTS):,} tokens) -> {out_path}")
