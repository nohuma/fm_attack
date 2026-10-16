#!/bin/bash
# Clean success rates of pi0 on the four reported LIBERO suites.
# Suites run one after another on the single GPU.
# Invoking this file backgrounds that sequence and returns immediately.

MODEL_NAME=pi0_libero_finetuned
BASE=/root/autodl-tmp/pi0_libero_finetuned
N_EPISODES=50
SEED=0
IMAGE_SIZE=256
CLEAN_ONLY=True
LOG_DIR=/root/fm_attack/log
PYTHON=/root/miniconda3/envs/vla/bin/python
SUITES=(libero_spatial libero_object libero_goal libero_10)

mkdir -p "$LOG_DIR"
DRIVER="$LOG_DIR/eval_${MODEL_NAME}_ep${N_EPISODES}_seed${SEED}_img${IMAGE_SIZE}_clean_driver.log"

if [[ "${1:-}" != "--foreground" ]]; then
    nohup bash "$0" --foreground >> "$DRIVER" 2>&1 &
    echo "pid $!"
    echo "driver $DRIVER"
    exit 0
fi

echo "model=$MODEL_NAME"
echo "base=$BASE"
echo "n_episodes=$N_EPISODES"
echo "clean_only=$CLEAN_ONLY"
echo "seed=$SEED"
echo "image_size=$IMAGE_SIZE"
echo "suites=${SUITES[*]}"
echo "started=$(date -Is)"

for suite in "${SUITES[@]}"; do
    tag="eval_${MODEL_NAME}_${suite}_ep${N_EPISODES}_seed${SEED}_img${IMAGE_SIZE}_clean"
    log="$LOG_DIR/${tag}.log"
    out="$LOG_DIR/${tag}.json"
    (
        echo "model=$MODEL_NAME"
        echo "suite=$suite"
        echo "base=$BASE"
        echo "n_episodes=$N_EPISODES"
        echo "clean_only=$CLEAN_ONLY"
        echo "seed=$SEED"
        echo "image_size=$IMAGE_SIZE"
        echo "started=$(date -Is)"
        PYTHONPATH=/root/fm_attack "$PYTHON" /root/fm_attack/eval/eval_libero.py \
            --base "$BASE" \
            --clean_only "$CLEAN_ONLY" \
            --suites "$suite" \
            --n_episodes "$N_EPISODES" \
            --seed "$SEED" \
            --image_size "$IMAGE_SIZE" \
            --output "$out"
        status=$?
        echo "exit=$status"
        echo "finished=$(date -Is)"
        echo "json=$out"
        exit "$status"
    ) > "$log" 2>&1
    echo "suite=$suite exit=$? log=$log json=$out"
done

echo "finished=$(date -Is)"
