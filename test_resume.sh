#!/bin/bash
# Test: crash-resume works correctly
set -e
cd "$(dirname "$0")"
rm -f checkpoints/latest.pt checkpoints/best.pt test_resume.log

echo "=== Step 1: Fresh start, train to S600 ==="
python3 -u train.py --batch_size 2 --max_steps 600 --save_every 5000 --amp --fresh > test_resume.log 2>&1

echo "=== Step 2: Kill process simulation - check checkpoint exists ==="
CKPT_STEP=$(python3 -c "import torch; ck=torch.load('checkpoints/latest.pt',map_location='cpu',weights_only=False); print(ck['step'])")
echo "  Checkpoint at step: $CKPT_STEP"
[ "$CKPT_STEP" -ge 200 ] || { echo "FAIL: no checkpoint"; exit 1; }

echo "=== Step 3: Resume (crash recovery) ==="
python3 -u train.py --batch_size 2 --max_steps 1000 --save_every 5000 --amp >> test_resume.log 2>&1

echo "=== Step 4: Verify ==="
grep -c "Resumed at step" test_resume.log
STEPS=$(grep -oP 'Done \K[0-9]+' test_resume.log)
FINAL=$(echo "$STEPS" | tail -1)
echo "  Final step: $FINAL"
[ "$FINAL" -ge 1000 ] && echo "PASS: resumed and completed $FINAL steps" || { echo "FAIL"; exit 1; }
