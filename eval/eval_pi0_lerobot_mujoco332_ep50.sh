#!/bin/bash
# Official lerobot eval for pi0. MuJoCo is 3.3.2.
# No image size override, so LIBERO renders at its default 360.
# Rollout uses n_action_steps=10. Spatial and object each get their own folder.
# n_episodes=50 per task. Running this file backgrounds the job and returns immediately.

MODEL_NAME=pi0_libero_finetuned
POLICY_PATH=/root/autodl-tmp/pi0_libero_finetuned
N_EPISODES=50
N_ACTION_STEPS=10
BATCH_SIZE=1
MAX_PARALLEL_TASKS=1
SUITES=(libero_spatial libero_object)
LOG_DIR=/root/fm_attack/eval_log
PYTHON=/root/miniconda3/envs/vla/bin/python
EVAL_PY=/root/miniconda3/envs/vla/lib/python3.12/site-packages/lerobot/scripts/lerobot_eval.py
DRIVER="$LOG_DIR/eval_${MODEL_NAME}_spatial_object_ep${N_EPISODES}_bs${BATCH_SIZE}_nact${N_ACTION_STEPS}_imgdefault_driver.log"

mkdir -p "$LOG_DIR"

if [[ "${1:-}" != "--foreground" ]]; then
    nohup bash "$0" --foreground >> "$DRIVER" 2>&1 &
    echo "pid $!"
    echo "driver $DRIVER"
    exit 0
fi

echo "model=$MODEL_NAME"
echo "policy_path=$POLICY_PATH"
echo "policy_family=pi0"
echo "tool=lerobot_eval.py"
echo "mujoco=3.3.2"
echo "suites=${SUITES[*]}"
echo "n_episodes=$N_EPISODES"
echo "n_action_steps=$N_ACTION_STEPS"
echo "batch_size=$BATCH_SIZE"
echo "max_parallel_tasks=$MAX_PARALLEL_TASKS"
echo "image_size=default_360"
echo "seed=default_1000"
echo "started=$(date -Is)"

for suite in "${SUITES[@]}"; do
    tag="eval_${MODEL_NAME}_${suite}_ep${N_EPISODES}_bs${BATCH_SIZE}_nact${N_ACTION_STEPS}_imgdefault"
    out="$LOG_DIR/$tag"
    echo "suite=$suite output_dir=$out started=$(date -Is)"
    "$PYTHON" "$EVAL_PY" \
        --policy.path="$POLICY_PATH" \
        --policy.n_action_steps="$N_ACTION_STEPS" \
        --env.type=libero \
        --env.task="$suite" \
        --env.max_parallel_tasks="$MAX_PARALLEL_TASKS" \
        --eval.batch_size="$BATCH_SIZE" \
        --eval.n_episodes="$N_EPISODES" \
        --output_dir="$out"
    status=$?
    echo "suite=$suite exit=$status finished=$(date -Is)"
    echo "eval_info=$out/eval_info.json"
    if [[ "$status" -ne 0 ]]; then
        echo "stopped after $suite"
        exit "$status"
    fi
done

echo "finished=$(date -Is)"
