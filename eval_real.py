"""
Real code eval: generate, compile, execute, and verify correctness.
"""
import torch, sys, os, re, io, contextlib
sys.path.insert(0, os.path.dirname(__file__))

from tiny_model import TinyConfig, TinyModel
from train import BPETok

TEST_CASES = [
    {
        "prompt": "def fibonacci(n):",
        "tests": [(5,), (0,), (1,)],
        "fn_name": "fibonacci",
        "expected": [5, 0, 1],
    },
    {
        "prompt": "def add(a, b):",
        "tests": [(1, 2), (5, 3), (0, 0)],
        "fn_name": "add",
        "expected": [3, 8, 0],
    },
    {
        "prompt": "def is_even(n):",
        "tests": [(2,), (3,), (0,)],
        "fn_name": "is_even",
        "expected": [True, False, True],
    },
    {
        "prompt": "def factorial(n):",
        "tests": [(0,), (1,), (5,)],
        "fn_name": "factorial",
        "expected": [1, 1, 120],
    },
    {
        "prompt": "def count_vowels(s):",
        "tests": [("hello",), ("sky",), ("AEIOU",)],
        "fn_name": "count_vowels",
        "expected": [2, 0, 5],
    },
    {
        "prompt": "def double(x):",
        "tests": [(3,), (0,), (-5,)],
        "fn_name": "double",
        "expected": [6, 0, -10],
    },
    {
        "prompt": "def square(x):",
        "tests": [(2,), (0,), (-3,)],
        "fn_name": "square",
        "expected": [4, 0, 9],
    },
    {
        "prompt": "def max_of_two(a, b):",
        "tests": [(1, 2), (5, 3), (0, 0)],
        "fn_name": "max_of_two",
        "expected": [2, 5, 0],
    },
]

def eval_one(model, tok, device, case, max_tokens=150, temp=0.2):
    """Generate code, compile, run test cases. Returns (passed, output_text, detail)."""
    ids = torch.tensor([tok.encode_prompt(case["prompt"])]).to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_tokens, n_loops=4, temperature=temp, top_k=40)
    text = tok.decode(out[0].tolist())

    # Extract function body
    fn_name = case["fn_name"]
    pattern = rf'(def\s+{fn_name}\s*\([^)]*\)\s*:[^\n]*(?:\n[ \t]+[^\n]*)*)'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        # Fallback: just the def line + pass
        pattern2 = rf'(def\s+{fn_name}\s*\([^)]*\)\s*:[^\n]*)'
        match = re.search(pattern2, text)
        if not match:
            return False, text, "no valid def found"

    code = match.group(1)

    # Close incomplete blocks
    lines = code.split('\n')
    if len(lines) == 1 or (len(lines) > 1 and not lines[-1].startswith(' ')):
        code = code + '\n    pass'

    # Strip unterminated triple-quote docstrings
    triple_count = code.count('"""') + code.count("'''")
    if triple_count % 2 != 0:
        # Odd number of triple quotes — close them
        code = code.rstrip() + '\n    pass\n"""'

    # Try to compile
    try:
        compile(code, "<eval>", "exec")
    except SyntaxError as e:
        return False, text, f"syntax error: {e}"

    # Execute and run tests
    namespace = {}
    try:
        exec(code, namespace)
    except Exception as e:
        return False, text, f"exec error: {e}"

    fn = namespace.get(fn_name)
    if fn is None:
        return False, text, f"function {fn_name} not found"

    results = []
    for args, expected in zip(case["tests"], case["expected"]):
        try:
            result = fn(*args) if isinstance(args, tuple) else fn(args)
            results.append(result == expected)
        except Exception as e:
            results.append(False)

    passed_count = sum(results)
    total = len(results)
    if passed_count == total:
        return True, text, f"all {total} tests passed"
    else:
        return False, text, f"{passed_count}/{total} tests passed"


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = TinyConfig()
    model = TinyModel(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    tok = BPETok(8192, "tokenizer.json")

    print(f"Model: {model.num_parameters():,} params | Device: {device}")
    print(f"Checkpoint: {ckpt_path}\n")

    passed = 0
    for case in TEST_CASES:
        ok, text, msg = eval_one(model, tok, device, case)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case['fn_name']}: {msg}")
        if not ok:
            snippet = text[:120].replace("\n", "\\n")
            print(f"  output: {snippet}")
        else:
            passed += 1

    print(f"\nResult: {passed}/{len(TEST_CASES)} passed")


if __name__ == "__main__":
    main()
