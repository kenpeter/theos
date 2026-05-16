#!/bin/bash
# Robust auto-restart training — NEVER stops until max_steps reached
# Usage: bash run.sh

MAX_RETRIES=100
RETRY=0

while true; do
    echo "[$(date)] Starting/resuming training (attempt $((RETRY+1))/$MAX_RETRIES)..."

    python3 -u train.py \
        --batch_size 2 \
        --max_steps 350000 \
        --save_every 5000 \
        --amp \
        --resume 2>&1

    EXIT=$?

    # Check if actually done (reached max_steps)
    LAST_STEP=$(python3 -c "import torch; c=torch.load('checkpoints/latest.pt',map_location='cpu',weights_only=False); print(c['step'])" 2>/dev/null)
    if [ "$LAST_STEP" -ge 350000 ] 2>/dev/null; then
        echo "[$(date)] Reached max_steps ($LAST_STEP/350000). Done."
        break
    fi

    RETRY=$((RETRY + 1))
    if [ "$RETRY" -ge "$MAX_RETRIES" ]; then
        echo "[$(date)] Max retries reached ($MAX_RETRIES). Giving up."
        break
    fi

    echo "[$(date)] Exited $EXIT (step=$LAST_STEP), restarting in 3s..."
    sleep 3
done
