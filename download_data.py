"""
Download datasets (wiki + code + fineweb) and save locally for fast training.
"""
import os, json, pickle
from pathlib import Path


def download():
    from datasets import load_dataset
    save_dir = Path("dataset_cache")
    save_dir.mkdir(exist_ok=True)

    # 1. Wiki (all ~36K texts)
    wiki_path = save_dir / "wiki_texts.pkl"
    if not wiki_path.exists():
        print("Downloading WikiText...")
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train", streaming=False)
        texts = [r["text"] for r in ds if len(r["text"].strip()) >= 50]
        with open(wiki_path, "wb") as f:
            pickle.dump(texts, f)
        print(f"  Saved {len(texts)} wiki texts")
    else:
        with open(wiki_path, "rb") as f:
            texts = pickle.load(f)
        print(f"  Wiki: {len(texts)} texts (cached)")

    # 2. CodeAlpaca (all 20K)
    code_path = save_dir / "code_texts.pkl"
    if not code_path.exists():
        print("Downloading CodeAlpaca...")
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train", streaming=False)
        texts = []
        for r in ds:
            t = f"<|user|>\n{r['instruction']}\n### Input:\n{r['input']}\n<|assistant|>\n{r['output']}\n<|end|>\n"
            if len(t.strip()) >= 50:
                texts.append(t)
        with open(code_path, "wb") as f:
            pickle.dump(texts, f)
        print(f"  Saved {len(texts)} code texts")
    else:
        with open(code_path, "rb") as f:
            texts = pickle.load(f)
        print(f"  Code: {len(texts)} texts (cached)")

    # 3. FineWeb sample (download first 100K)
    web_path = save_dir / "web_texts.pkl"
    if not web_path.exists():
        print("Downloading FineWeb (100K texts, this may take a while)...")
        ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
        texts = []
        for i, r in enumerate(ds):
            if i >= 100000:
                break
            t = r.get("text", "")
            if len(t) >= 100:
                texts.append(t)
            if i > 0 and i % 20000 == 0:
                print(f"  Downloaded {i}...")
        with open(web_path, "wb") as f:
            pickle.dump(texts, f)
        print(f"  Saved {len(texts)} web texts")
    else:
        with open(web_path, "rb") as f:
            texts = pickle.load(f)
        print(f"  Web: {len(texts)} texts (cached)")

    print("\nAll datasets downloaded and cached!")


if __name__ == "__main__":
    download()