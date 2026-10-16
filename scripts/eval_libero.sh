nohup python /root/fm_attack/eval/eval_libero.py \
    --base /root/autodl-tmp/pi0_libero_finetuned \
    --checkpoints ["/root/autodl-tmp/outputs/stage1/step_5000,/path/to/stage2/step_30000"] \
    --suites libero_spatial \
    --n_episodes 1 \
    --output /root/fm_attack/log/eval_libero.json \
    > /root/fm_attack/log/eval_libero.log 2>&1 &
