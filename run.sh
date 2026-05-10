#!/bin/bash
# Auto-restart training — resumes on any failure
# Usage: bash run.sh [extra train.py args...]

while true; do
    echo "[$(date)] Starting/resuming training..."

    python3 -u train.py \
        --batch_size 2 \
        --max_steps 350000 \
        --save_every 5000 \
        --amp \
        --resume \
        "$@" 2>&1

    EXIT=$?
    if [ "$EXIT" -eq 0 ]; then
        echo "[$(date)] Completed normally."
        break
    fi

    echo "[$(date)] Exited $EXIT, restarting in 2s..."
    sleep 2
done
