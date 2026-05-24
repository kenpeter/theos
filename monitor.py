"""
Monitor training progress: GPU, checkpoints, eval.
"""
import subprocess, json, os, time, sys, torch

CHECK_INTERVAL = 120
EVAL_EVERY = 6
METRICS_FILE = "training_metrics.json"

def gpu_info():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        return r.stdout.strip()
    except: return "N/A"

def checkpoint_step(path):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        return ckpt.get("step", "?")
    except: return "?"

def run_eval(path):
    try:
        r = subprocess.run(
            ["python3", "eval_real.py", path],
            capture_output=True, text=True, timeout=130
        )
        _passes = r.stdout.count("[PASS]")
        _fails = r.stdout.count("[FAIL]")
        _total = _passes + _fails
        _compile_ok = r.stdout.count("syntax error") + r.stdout.count("no valid def")
        _compile_acc = max(0, (_total - _compile_ok)) / max(1, _total)
        return _passes, _total, _compile_acc, r.stdout.strip().split("\n")[-3:]
    except Exception as e:
        return None, None, None, str(e)

def last_metrics():
    try:
        with open(METRICS_FILE) as f:
            data = json.load(f)
        if data:
            return data[-1]
    except: pass
    return None

print("=" * 50)
print("Training Monitor Started")
print(f"PID: {os.getpid()}")
print("=" * 50)

last_best_mtime = 0
last_latest_mtime = 0
eval_count = 0

try:
    while True:
        gpu = gpu_info()
        best_step = checkpoint_step("checkpoints/best.pt")
        latest_step = checkpoint_step("checkpoints/latest.pt")
        lm = last_metrics()
        best_mtime = os.path.getmtime("checkpoints/best.pt")
        latest_mtime = os.path.getmtime("checkpoints/latest.pt")

        now = time.strftime("%H:%M:%S")
        line = f"[{now}] GPU: {gpu} | best.pt: step={best_step} | latest.pt: step={latest_step}"
        if lm:
            line += f" | last_log: step={lm['step']} train_loss={lm['train_loss']:.4f} val_loss={lm['val_loss']:.4f}"

        if latest_mtime != last_latest_mtime:
            line += " [training active]"
        else:
            line += " [IDLE - check if training is stuck]"

        print(line, flush=True)

        if best_mtime != last_best_mtime:
            print(f"  ★ best.pt updated! Running eval on best.pt...", flush=True)
            passes, total, compile_acc, last_lines = run_eval("checkpoints/best.pt")
            if passes is not None:
                status = "GREEN" if passes > 0 else ("YELLOW" if compile_acc >= 0.10 else "RED")
                print(f"  ★ EVAL: {passes}/{total} passed | compile: {compile_acc:.0%} | status: {status}", flush=True)
                if last_lines:
                    for l in last_lines: print(f"    {l}", flush=True)
            else:
                print(f"  ★ EVAL error: {last_lines}", flush=True)

        if best_mtime == last_best_mtime:
            eval_count += 1
        else:
            eval_count = 0
        if eval_count >= EVAL_EVERY:
            eval_count = 0
            print(f"  Periodic eval on latest.pt...", flush=True)
            passes, total, compile_acc, last_lines = run_eval("checkpoints/latest.pt")
            if passes is not None:
                status = "GREEN" if passes > 0 else ("YELLOW" if compile_acc >= 0.10 else "RED")
                print(f"  EVAL(latest): {passes}/{total} passed | compile: {compile_acc:.0%} | status: {status}", flush=True)
                if last_lines:
                    for l in last_lines: print(f"    {l}", flush=True)

        last_best_mtime = best_mtime
        last_latest_mtime = latest_mtime

        time.sleep(CHECK_INTERVAL)
except KeyboardInterrupt:
    print("\nMonitor stopped.")
