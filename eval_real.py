"""
Real code eval: generate, compile, execute, and verify correctness.
"""
import torch, sys, os, re, io, contextlib
sys.path.insert(0, os.path.dirname(__file__))

from tiny_model import TinyConfig, TinyModel
from train import BPETok

TEST_CASES = [
    {
        "prompt": "<|user|>\nWrite a Python function `fibonacci(n)` that returns the nth Fibonacci number.\n<|assistant|>\ndef fibonacci(n):",
        "bare_prompt": "def fibonacci(n):",
        "tests": [(5,), (0,), (1,)],
        "fn_name": "fibonacci",
        "expected": [5, 0, 1],
    },
    {
        "prompt": "<|user|>\nWrite a Python function `add(a, b)` that returns the sum of a and b.\n<|assistant|>\ndef add(a, b):",
        "bare_prompt": "def add(a, b):",
        "tests": [(1, 2), (5, 3), (0, 0)],
        "fn_name": "add",
        "expected": [3, 8, 0],
    },
    {
        "prompt": "<|user|>\nWrite a Python function `is_even(n)` that returns True if n is even, False otherwise.\n<|assistant|>\ndef is_even(n):",
        "bare_prompt": "def is_even(n):",
        "tests": [(2,), (3,), (0,)],
        "fn_name": "is_even",
        "expected": [True, False, True],
    },
    {
        "prompt": "<|user|>\nWrite a Python function `factorial(n)` that returns n factorial (n!).\n<|assistant|>\ndef factorial(n):",
        "bare_prompt": "def factorial(n):",
        "tests": [(0,), (1,), (5,)],
        "fn_name": "factorial",
        "expected": [1, 1, 120],
    },
    {
        "prompt": "<|user|>\nWrite a Python function `count_vowels(s)` that counts vowels (a,e,i,o,u) in a string.\n<|assistant|>\ndef count_vowels(s):",
        "bare_prompt": "def count_vowels(s):",
        "tests": [("hello",), ("sky",), ("AEIOU",)],
        "fn_name": "count_vowels",
        "expected": [2, 0, 5],
    },
    {
        "prompt": "<|user|>\nWrite a Python function `double(x)` that returns x multiplied by 2.\n<|assistant|>\ndef double(x):",
        "bare_prompt": "def double(x):",
        "tests": [(3,), (0,), (-5,)],
        "fn_name": "double",
        "expected": [6, 0, -10],
    },
    {
        "prompt": "<|user|>\nWrite a Python function `square(x)` that returns x squared (x * x).\n<|assistant|>\ndef square(x):",
        "bare_prompt": "def square(x):",
        "tests": [(2,), (0,), (-3,)],
        "fn_name": "square",
        "expected": [4, 0, 9],
    },
    {
        "prompt": "<|user|>\nWrite a Python function `max_of_two(a, b)` that returns the larger of a and b.\n<|assistant|>\ndef max_of_two(a, b):",
        "bare_prompt": "def max_of_two(a, b):",
        "tests": [(1, 2), (5, 3), (0, 0)],
        "fn_name": "max_of_two",
        "expected": [2, 5, 0],
    },
]

def eval_one(model, tok, device, case, max_tokens=200, temp=0.1):
    """Generate code, compile, run test cases. Returns (passed, output_text, detail)."""
    ids = torch.tensor([tok.encode_prompt(case["prompt"])]).to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_tokens, n_loops=4, temperature=temp, top_k=20)
    text = tok.decode(out[0].tolist())

    # Try chat format; if no match, try bare prompt format
    fn_name = case["fn_name"]
    best_code = None
    for prompt_key in ["prompt", "bare_prompt"]:
        prompt_text = case.get(prompt_key)
        if not prompt_text:
            continue
        p_ids = torch.tensor([tok.encode_prompt(prompt_text)]).to(device)
        with torch.no_grad():
            p_out = model.generate(p_ids, max_new_tokens=max_tokens, n_loops=4, temperature=temp, top_k=20)
        p_text = tok.decode(p_out[0].tolist())
        match = re.search(rf'(def\s+{fn_name}\s*\([^)]*\)\s*:[^\n]*(?:\n[ \t]+[^\n]*)*)', p_text, re.DOTALL)
        if match:
            best_code = match.group(1)
            text = p_text
            break

    if not best_code:
        match = re.search(rf'(def\s+{fn_name}\s*\([^)]*\)\s*:[^\n]*)', text)
        if not match:
            return False, text, "no valid def found"
        best_code = match.group(1)

    code = best_code
    # Close incomplete blocks
    lines = code.split('\n')
    if len(lines) == 1 or (len(lines) > 1 and not lines[-1].startswith(' ') and lines[-1].strip() != 'return'):
        last_line = lines[-1].strip()
        if not (last_line.startswith('return') or last_line.startswith('if') or last_line.startswith('for') or last_line.startswith('while')):
            code = code.rstrip()

    # Strip unterminated triple-quote docstrings
    triple_count = code.count('"""') + code.count("'''")
    if triple_count % 2 != 0:
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
