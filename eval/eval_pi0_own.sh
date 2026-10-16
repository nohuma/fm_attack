#!/bin/bash
# Clean pi0 success rates with our own eval_libero.py.
# spatial and object only. MuJoCo is 3.3.0.
# Images stay 256. The checkpoint stores n_action_steps=50; this run uses 10.
# One environment, so this matches batch_size=1 and max_parallel_tasks=1.
# Running this file backgrounds the sequence and returns immediately.

MODEL_NAME=pi0_libero_finetuned
BASE=/root/autodl-tmp/pi0_libero_finetuned
N_EPISODES=10
N_ACTION_STEPS=10
SEED=1000
IMAGE_SIZE=256
DEVICE=cuda
LOG_DIR=/root/fm_attack/log/own_eval_pi0_mujoco330_nact10
PYTHON=/root/miniconda3/envs/vla/bin/python
SUITES=(libero_spatial libero_object)

mkdir -p "$LOG_DIR"
DRIVER="$LOG_DIR/eval_${MODEL_NAME}_own_ep${N_EPISODES}_seed${SEED}_img${IMAGE_SIZE}_nact${N_ACTION_STEPS}_driver.log"

if [[ "${1:-}" != "--foreground" ]]; then
    nohup bash "$0" --foreground >> "$DRIVER" 2>&1 &
    echo "pid $!"
    echo "driver $DRIVER"
    exit 0
fi

echo "model=$MODEL_NAME"
echo "policy_path=$BASE"
echo "policy_family=pi0"
echo "tool=eval_libero"
echo "mujoco=3.3.0"
echo "n_episodes=$N_EPISODES"
echo "n_action_steps=$N_ACTION_STEPS"
echo "seed=$SEED"
echo "image_size=$IMAGE_SIZE"
echo "clean_only=true"
echo "device=$DEVICE"
echo "suites=${SUITES[*]}"
echo "started=$(date -Is)"

for suite in "${SUITES[@]}"; do
    tag="eval_${MODEL_NAME}_own_${suite}_ep${N_EPISODES}_seed${SEED}_img${IMAGE_SIZE}_nact${N_ACTION_STEPS}"
    log="$LOG_DIR/${tag}.log"
    out="$LOG_DIR/${tag}.json"
    (
        echo "model=$MODEL_NAME"
        echo "policy_path=$BASE"
        echo "policy_family=pi0"
        echo "suite=$suite"
        echo "tool=eval_libero"
        echo "mujoco=3.3.0"
        echo "n_episodes=$N_EPISODES"
        echo "n_action_steps=$N_ACTION_STEPS"
        echo "seed=$SEED"
        echo "image_size=$IMAGE_SIZE"
        echo "clean_only=true"
        echo "device=$DEVICE"
        echo "output=$out"
        echo "started=$(date -Is)"
        PYTHONPATH=/root/fm_attack "$PYTHON" /root/fm_attack/eval/eval_libero.py \
            --base "$BASE" \
            --clean_only True \
            --suites "$suite" \
            --n_episodes "$N_EPISODES" \
            --n_action_steps "$N_ACTION_STEPS" \
            --seed "$SEED" \
            --image_size "$IMAGE_SIZE" \
            --device "$DEVICE" \
            --output "$out"
        status=$?
        echo "exit=$status"
        echo "finished=$(date -Is)"
        echo "output=$out"
        exit "$status"
    ) > "$log" 2>&1
    echo "suite=$suite exit=$? log=$log output=$out"
done

echo "finished=$(date -Is)"
