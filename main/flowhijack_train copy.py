"""Train a FlowHijack-poisoned policy. Recovery loads the merged checkpoint this writes."""

import draccus
import torch

from configs import BDConfig
from methods.flowhijack import FlowHijack, flowhijack_views
from triggers import build_trigger
from utils.data import make_lerobot_dataloader
from utils.model import load_policy_and_preprocess
from utils.training import MetricLogger, build_scheduler, experiment_dir, make_optimizer, run_training, set_seed


@draccus.wrap()
def main(cfg: BDConfig) -> None:
    set_seed(cfg.training.seed)
    output_dir = experiment_dir(cfg)
    print("=" * 40)
    print(f"output {output_dir}")

    attack_cfg = cfg.attack
    assert attack_cfg.name == "flowhijack", f"flowhijack_train.py only runs flowhijack, got {attack_cfg.name}"
    attack = FlowHijack(
        tau0=attack_cfg.tau0,
        loss_weights=attack_cfg.loss_weights,
        target_action=attack_cfg.target_action,
        action_delta=attack_cfg.action_delta,
        lora=attack_cfg.lora,
    )

    trigger = build_trigger(
        cfg.data.trigger,
        camera_keys=cfg.model.camera_keys,
        size=cfg.data.trigger_size,
        position=cfg.data.trigger_position,
        color=cfg.data.trigger_color,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, preprocess = load_policy_and_preprocess(
        cfg.model.model_path_or_name, cfg.model.processor_path, device
    )
    policy = attack.attach_lora(policy)
    policy.train()

    trainable_params = sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad) # 统计训练参数量
    print("=" * 40)
    print(f"trainable params {trainable_params}")
    print("=" * 40)

    dataloader, dataset, poison_episodes = make_lerobot_dataloader(
        policy,
        repo_id=cfg.data.dataset_repo_id,
        root=cfg.data.dataset_dir,
        fps=cfg.data.dataset_fps,
        poison_ratio=cfg.data.poison_ratio,
        batch_size=cfg.training.batch_size,
        seed=cfg.training.seed,
        pin_memory=device.type == "cuda",
    )
    optimizer = make_optimizer(policy, cfg.training.learning_rate)
    scheduler = build_scheduler(cfg, optimizer)

    run_training(
        policy,
        dataloader,
        attack,
        optimizer,
        dataset,
        preprocess,
        cfg.model.camera_keys,
        lambda batch: flowhijack_views(batch, poison_episodes, trigger),
        max_steps=cfg.training.max_steps,
        grad_accumulation_steps=cfg.training.grad_accumulation_steps,
        output_dir=output_dir,
        save_freq=cfg.logging.save_freq,
        save_latest_only=cfg.logging.save_latest_checkpoint_only,
        use_lora=True,
        merge_lora=cfg.attack.merge_lora,
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
