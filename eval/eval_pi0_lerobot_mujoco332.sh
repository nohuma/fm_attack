#!/bin/bash
# Official lerobot eval for pi0. MuJoCo is 3.3.2.
# No image size override, so LIBERO renders at its default 360.
# Rollout uses n_action_steps=10. Only spatial and object.
# Running this file backgrounds the job and returns immediately.

MODEL_NAME=pi0_libero_finetuned
POLICY_PATH=/root/autodl-tmp/pi0_libero_finetuned
N_EPISODES=10
N_ACTION_STEPS=10
BATCH_SIZE=1
MAX_PARALLEL_TASKS=1
TASKS=libero_spatial,libero_object
LOG_DIR=/root/fm_attack/log/lerobot_eval_mujoco332_nact10_imgdefault
PYTHON=/root/miniconda3/envs/vla/bin/python
EVAL_PY=/root/miniconda3/envs/vla/lib/python3.12/site-packages/lerobot/scripts/lerobot_eval.py
TAG="eval_${MODEL_NAME}_spatial_object_ep${N_EPISODES}_bs${BATCH_SIZE}_nact${N_ACTION_STEPS}_imgdefault"

mkdir -p "$LOG_DIR"
DRIVER="$LOG_DIR/${TAG}_driver.log"
OUT="$LOG_DIR/$TAG"

if [[ "${1:-}" != "--foreground" ]]; then
    nohup bash "$0" --foreground >> "$DRIVER" 2>&1 &
    echo "pid $!"
    echo "log $DRIVER"
    echo "output_dir $OUT"
    exit 0
fi

echo "model=$MODEL_NAME"
echo "policy_path=$POLICY_PATH"
echo "policy_family=pi0"
echo "tool=lerobot_eval.py"
echo "mujoco=3.3.2"
echo "tasks=$TASKS"
echo "n_episodes=$N_EPISODES"
echo "n_action_steps=$N_ACTION_STEPS"
echo "batch_size=$BATCH_SIZE"
echo "max_parallel_tasks=$MAX_PARALLEL_TASKS"
echo "image_size=default_360"
echo "seed=default_1000"
echo "output_dir=$OUT"
echo "started=$(date -Is)"

"$PYTHON" "$EVAL_PY" \
    --policy.path="$POLICY_PATH" \
    --policy.n_action_steps="$N_ACTION_STEPS" \
    --env.type=libero \
    --env.task="$TASKS" \
    --env.max_parallel_tasks="$MAX_PARALLEL_TASKS" \
    --eval.batch_size="$BATCH_SIZE" \
    --eval.n_episodes="$N_EPISODES" \
    --output_dir="$OUT"
status=$?
echo "exit=$status"
echo "finished=$(date -Is)"
echo "eval_info=$OUT/eval_info.json"
exit "$status"
