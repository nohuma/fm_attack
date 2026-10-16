PYTHONPATH=/root/fm_attack nohup /root/miniconda3/envs/vla/bin/python /root/fm_attack/eval/eval_libero.py \
    --base /root/autodl-tmp/pi0_libero_finetuned \
    --clean_only True \
    --suites libero_spatial \
    --n_episodes 1 \
    --output /root/fm_attack/log/eval_pi0_clean.json \
    > /root/fm_attack/log/eval_pi0_clean.log 2>&1 &


PYTHONPATH=/root/fm_attack nohup /root/miniconda3/envs/vla/bin/python /root/fm_attack/eval/eval_libero.py \
    --base /root/autodl-tmp/pi0_libero_finetuned \
    --clean_only True \
    --suites libero_spatial \
    --task_ids [0] \
    --n_episodes 1 \
    --output /root/fm_attack/log/eval_pi0_clean.json \
    > /root/fm_attack/log/eval_pi0_clean.log 2>&1 &
