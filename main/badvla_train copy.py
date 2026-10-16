"""Experiment entry: assemble a BadVLA run and hand it to the shared training loop."""

import draccus
import torch

from configs import BDConfig
from methods.badvla import BadVLA, badvla_views
from triggers import build_trigger
from utils.checkpoint import load_adapter_or_checkpoint
from utils.data import make_lerobot_dataloader
from utils.model import load_policy_and_preprocess
from utils.training import MetricLogger, build_scheduler, experiment_dir, make_optimizer, run_training, set_seed


@draccus.wrap()
def main(cfg: BDConfig) -> None:
    set_seed(cfg.training.seed)
    output_dir = experiment_dir(cfg)
    print("=" * 40)
    print(f"output: {output_dir}")
    # =============================加载攻击、触发器、方法、数据集（投毒）================================
    attack_cfg = cfg.attack
    assert attack_cfg.name == "badvla", f"badvla_train.py only runs badvla, got {attack_cfg.name}"
    attack = BadVLA(
        stage=attack_cfg.stage,
        loss_weights=attack_cfg.loss_weights,
        lora=attack_cfg.lora,
    )

    trigger = build_trigger(
        cfg.data.trigger,
        camera_keys=cfg.model.camera_keys,
        size=cfg.data.trigger_size, # 0.1
        position=cfg.data.trigger_position,
        color=cfg.data.trigger_color,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, preprocess = load_policy_and_preprocess(
        cfg.model.model_path_or_name, cfg.model.processor_path, device
    )
    if attack.stage == 2 and attack_cfg.stage1_path:
        policy = load_adapter_or_checkpoint(policy, attack_cfg.stage1_path)
    elif attack.stage == 2 and not attack_cfg.stage1_path:
        raise ValueError("You need to get a model trained with stage 1.")

    policy = attack.attach_lora(policy)
    policy.train()
    trainable_params = sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad) # 统计训练参数量
    print("=" * 40)
    print(f"{attack.name} stage {attack.stage}  trainable params {trainable_params}")
    print("=" * 40)

    dataloader, dataset, poison_episodes = make_lerobot_dataloader(
        policy,
        repo_id=cfg.data.dataset_repo_id,
        root=cfg.data.dataset_dir,
        fps=cfg.data.dataset_fps,
        poison_ratio=cfg.data.poison_ratio, # 0,1
        batch_size=cfg.training.batch_size, # stage 1: 8, stage 2: 16
        seed=cfg.training.seed,
        pin_memory=device.type == "cuda",
        poison_only=attack.stage == 1, # 第一阶段就只采样中毒轨迹，第二阶段采样全部轨迹
    )

    optimizer = make_optimizer(policy, cfg.training.learning_rate)
    scheduler = build_scheduler(cfg, optimizer)
    # ===========================正式训练================================
    run_training(
        policy,                                                                             # 模型
        dataloader,                                                                         # 数据
        attack,                                                                             # 攻击方法、lora；损失
        optimizer,                                                                          # 训练
        dataset,                                                                            # 辅助信息
        preprocess,                                                                         # 输入编码器
        cfg.model.camera_keys,                                                              # 输入处理
        lambda batch: badvla_views(batch, attack.stage, poison_episodes, trigger),
        max_steps=cfg.training.max_steps,
        grad_accumulation_steps=cfg.training.grad_accumulation_steps,
        output_dir=output_dir,
        save_freq=cfg.logging.save_freq,
        save_latest_only=cfg.logging.save_latest_checkpoint_only,
        use_lora=True,                                                                      # 默认必须采用lora做微调
        merge_lora=attack_cfg.merge_lora,
        scheduler=scheduler,
        logger=MetricLogger(
            output_dir,
            wandb_entity=cfg.logging.wandb_entity,
            wandb_project=cfg.logging.wandb_project,
            log_freq=cfg.logging.wandb_log_freq,
        ),
    )


if __name__ == "__main__":
    main()
