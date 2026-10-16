#!/bin/bash
# Clean pi0 success rates via the stock lerobot-eval CLI.
# Four reported suites, one after another. MuJoCo is pinned to 3.3.0.
# Images stay 256 to match the checkpoint. batch_size=1 avoids the auto
# multi-env default on this 14-core machine.
# Running this file backgrounds the sequence and returns immediately.

MODEL_NAME=pi0_libero_finetuned
POLICY_PATH=/root/autodl-tmp/pi0_libero_finetuned
N_EPISODES=10
SEED=1000
IMAGE_SIZE=256
BATCH_SIZE=1
N_ACTION_STEPS=10
DEVICE=cuda
LOG_DIR=/root/fm_attack/log/lerobot_eval_mujoco330_nact10
PYTHON=/root/miniconda3/envs/vla/bin/lerobot-eval
SUITES=(libero_spatial libero_object)

mkdir -p "$LOG_DIR"
DRIVER="$LOG_DIR/eval_${MODEL_NAME}_ep${N_EPISODES}_seed${SEED}_img${IMAGE_SIZE}_bs${BATCH_SIZE}_nact${N_ACTION_STEPS}_driver.log"

if [[ "${1:-}" != "--foreground" ]]; then
    nohup bash "$0" --foreground >> "$DRIVER" 2>&1 &
    echo "pid $!"
    echo "driver $DRIVER"
    exit 0
fi

echo "model=$MODEL_NAME"
echo "policy_path=$POLICY_PATH"
echo "tool=lerobot-eval"
echo "mujoco=3.3.0"
echo "n_episodes=$N_EPISODES"
echo "seed=$SEED"
echo "image_size=$IMAGE_SIZE"
echo "batch_size=$BATCH_SIZE"
echo "n_action_steps=$N_ACTION_STEPS"
echo "device=$DEVICE"
echo "suites=${SUITES[*]}"
echo "started=$(date -Is)"

for suite in "${SUITES[@]}"; do
    tag="eval_${MODEL_NAME}_${suite}_ep${N_EPISODES}_seed${SEED}_img${IMAGE_SIZE}_bs${BATCH_SIZE}_nact${N_ACTION_STEPS}"
    log="$LOG_DIR/${tag}.log"
    out="$LOG_DIR/${tag}"
    (
        echo "model=$MODEL_NAME"
        echo "suite=$suite"
        echo "policy_path=$POLICY_PATH"
        echo "tool=lerobot-eval"
        echo "mujoco=3.3.0"
        echo "n_episodes=$N_EPISODES"
        echo "seed=$SEED"
        echo "image_size=$IMAGE_SIZE"
        echo "batch_size=$BATCH_SIZE"
        echo "n_action_steps=$N_ACTION_STEPS"
        echo "device=$DEVICE"
        echo "output_dir=$out"
        echo "started=$(date -Is)"
        "$PYTHON" \
            --policy.path="$POLICY_PATH" \
            --policy.device="$DEVICE" \
            --env.type=libero \
            --env.task="$suite" \
            --env.observation_height="$IMAGE_SIZE" \
            --env.observation_width="$IMAGE_SIZE" \
            --eval.n_episodes="$N_EPISODES" \
            --eval.batch_size="$BATCH_SIZE" \
            --eval.use_async_envs=false \
            --policy.n_action_steps="$N_ACTION_STEPS" \
            --seed="$SEED" \
            --job_name="$tag" \
            --output_dir="$out"
        status=$?
        echo "exit=$status"
        echo "finished=$(date -Is)"
        echo "eval_info=$out/eval_info.json"
        exit "$status"
    ) > "$log" 2>&1
    echo "suite=$suite exit=$? log=$log output_dir=$out"
done

echo "finished=$(date -Is)"
