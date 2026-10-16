PYTHONPATH=/root/fm_attack nohup python /root/fm_attack/main/badvla_train.py \
    --config_path /root/fm_attack/configs/badvla_stage1.yaml \
    > /root/fm_attack/log/badvla_stage1_1.log 2>&1 &
tail -f /root/fm_attack/log/badvla_stage1_1.log

# 32位测试
PYTHONPATH=/root/fm_attack nohup python /root/fm_attack/main/badvla_train.py \
    --config_path /root/fm_attack/configs/badvla_stage1.yaml \
    > /root/fm_attack/log/badvla_stage1_2.log 2>&1 &


PYTHONPATH=/root/fm_attack nohup python /root/fm_attack/main/badvla_train.py \
    --config_path /root/fm_attack/configs/badvla_stage2.yaml \
    > /root/fm_attack/log/badvla_stage2.log 2>&1 &