"""LIBERO success rates for a BadVLA or FlowHijack policy.

Clean episodes use the simulator images. Attack episodes paint the same patch
used in training onto both cameras before the policy reads them. Both rates are
the fraction of episodes whose LIBERO task reports ``is_success``.

BadVLA checkpoints are two runs, stage 1 then stage 2. FlowHijack is one run.
A run directory that contains ``model.safetensors`` is loaded as a full model.
Otherwise its ``lora_adapter`` is merged in.

Flags follow the training scripts. A yaml is optional via ``--config_path``.

    python eval/eval_libero.py \\
        --base /root/autodl-tmp/pi0_libero_finetuned \\
        --checkpoints [path/to/stage1/step_5000,path/to/stage2/step_30000] \\
        --suites libero_spatial --n_episodes 1
"""

import json  
import sys  
from dataclasses import dataclass, field  
from pathlib import Path  

import draccus  
import numpy as np  
import torch  

# _ROOT = Path(__file__).resolve().parents[1]  # eval/eval_libero.py 的上一级就是 fm_attack
# if str(_ROOT) not in sys.path:  # 直接跑这个文件时，仓库根还不在 sys.path 里
#     sys.path.insert(0, str(_ROOT))  # 插到最前面，优先用本仓库的 eval、triggers、utils

from lerobot.envs.configs import LiberoEnv  # 套件、分辨率等配置；真正的 gym 环境由 create_envs 创建
from lerobot.envs.utils import preprocess_observation  # 仿真 numpy 图像转成 LeRobot 的 float 张量
from lerobot.policies.factory import make_pre_post_processors  
from lerobot.utils.constants import ACTION  # 动作在字典里的键名，后处理时用来取出动作张量

from eval.load_policy import load_for_eval  
from triggers import build_trigger  
from utils.model import unwrap_policy 


# 与 lerobot.envs.libero.TASK_SUITE_MAX_STEPS 的键一致；create_envs 也按这个名单建仿真
LEBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")


@dataclass
class EvalConfig:
    base: str = "/root/autodl-tmp/pi0_libero_finetuned"  # 原始 PI0 目录；完整合并权重也可以从这里加载
    checkpoints: list[str] = field(default_factory=list)  # 空着只加载原始 PI0。非空则按顺序合并 LoRA，和测不测触发器无关
    processor: str | None = None  # 归一化统计和分词器目录；空则用 base

    suites: str = "libero_spatial"  # 一个 LIBERO 套件名，例如 libero_spatial / libero_object / libero_goal / libero_10
    task_ids: list[int] | None = None  # 套件内要评的任务编号；None 表示该套件全部任务
    n_episodes: int = 1  # 每个任务重复多少个回合，换不同初始状态
    n_action_steps: int | None = None  # 覆盖策略一次执行的动作步数；空则用检查点里的值
    image_size: int = 256  # 仿真渲染边长，和训练时的 256 对齐，不用 LIBERO 默认的 360
    seed: int = 0  # 第 0 个回合的种子；后面的回合在此基础上加序号
    device: str = "cuda"  # 策略推理设备；没有 CUDA 时 evaluate 会退回 CPU FIXME:默认单卡运行

    camera_keys: list[str] = field(  # 贴色块的两路相机，键名和训练 prepare_batch 之后一致
        default_factory=lambda: ["observation.images.image", "observation.images.image2"]
    )
    trigger_type: str = 'patch'
    trigger_size: float = 0.1  # 色块边长占图像短边的比例，0.1 在 256 上大约 25 像素
    trigger_position: str = "center"  # 色块位置，和训练用的 PatchTrigger 相同
    trigger_color: float = 255  # 大于 1 时会在触发器里除以 255，变成图像上的 1.0

    clean_only: bool = False  # True 只测干净图像。False 时干净和贴色块都测，原始模型和 LoRA 都一样

    output: str | None = None  # 汇总 JSON 的保存路径；空则只打印到终端


def _episode_success(info: dict, n_envs: int) -> list[bool]:
    if "final_info" in info:  # 回合结束时，向量环境把该步的 info 放进 final_info
        final_info = info["final_info"]
        if isinstance(final_info, dict):  # 新版 gymnasium：各字段仍是长度为环境数的数组
            is_success = final_info.get("is_success", [False] * n_envs)  # 没有这个键就当成失败
            if hasattr(is_success, "tolist"):  # numpy 数组先拆成 Python 列表
                return [bool(item) for item in is_success.tolist()]
            return [bool(is_success)] * n_envs  # 标量则复制到每一个并行环境
        successes = []  # 旧版：final_info 是每个环境一个字典
        for item in final_info:
            if isinstance(item, dict) and "is_success" in item:  # 该环境这一步带了任务是否完成
                successes.append(bool(item["is_success"]))
            else:
                successes.append(False)  # 缺字段或还没结束，记为未成功
        return successes
    if "is_success" in info:  # 有的包装器在回合中途就把成功标志放在 info 顶层
        is_success = info["is_success"]
        if hasattr(is_success, "tolist"): # 可能有多个环境
            return [bool(item) for item in is_success.tolist()]
        return [bool(is_success)] * n_envs
    return [False] * n_envs  # 这一步没有成功信息，调用方会用 or 保住之前已经成功的记录


def rollout(
    env, policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor, trigger, seed: int | None
) -> bool:
    """One environment. ``trigger`` is applied after the policy preprocessor."""
    policy.reset()  # 清空动作队列，避免上一回合缓存的动作块被接着执行
    observation, info = env.reset(seed=[seed] if seed is not None else None)  # 向量环境的 reset 接收每个环境一个种子(批次); observation['pixels'][image, image2] 1 H W C； 这三个是 robot_state 里的三组机器人状态，都是 numpy 数组。eef 是机械臂末端。pos 是末端在空间中的三维位置，quat 是姿态四元数，mat 是同一个姿态的 3×3 旋转矩阵。gripper 是夹爪。qpos 是两个指头的开合位置，qvel 是它们的速度。joints 是手臂上的七个关节。pos 是七个关节角，vel 是七个关节速度。进策略之前，环境前处理不会把这三组都送进去。它只留末端位置、四元数转成的轴角、夹爪位置，拼成 8 维的 observation.state。
    done = False  # 当前回合是否已经终止或被截断
    max_steps = int(env.call("_max_episode_steps")[0])  # 向底层 LIBERO 取该套件的步数上限，只建了一个环境所以取第 0 个; 不同任务套件的最大上限可能不同
    success = False  # 一旦 BDDL 的 is_success 为真就保持为真
    step = 0  # 已执行的控制步数
    while not done and step < max_steps:  # 任务完成、超时，或达到套件步数上限就停
        observation = preprocess_observation(observation)  # uint8 HWC 转成 float32 CHW、范围 [0,1]，相机键加上 observation.images; 把数据处理成lebero的形式
        try:
            observation["task"] = list(env.call("task_description"))  # 取出该任务的语言指令，供 PI0 分词
        except (AttributeError, NotImplementedError):  # 环境没有这条指令时给空字符串，避免整次评测中断
            observation["task"] = [""]
        observation = env_preprocessor(observation)  # LIBERO 专用：图像翻转 180 度，机器人状态压成 8 维 observation.state
        observation = preprocessor(observation)  # PI0 处理器：状态归一化、任务分词，并放到策略所在设备
        if trigger is not None:  # 攻击回合才贴色块；干净回合传入 None
            observation = trigger(observation)  # 贴在策略预处理之后，和训练时 prepare_batch 之后再上色的顺序一致
        with torch.inference_mode():  # 评测不建计算图
            action = policy.select_action(observation)  # 动作队列空了才重新采样一个动作块，否则弹出下一步
        action = postprocessor(action)  # 把策略输出的归一化动作还原成环境要的数值
        action = env_postprocessor({ACTION: action})[ACTION]  # LIBERO 的环境后处理是空流水线，取出原动作
        action_numpy = action.to("cpu").numpy()  # 仿真步进要 CPU 上的 numpy
        observation, _, terminated, truncated, info = env.step(action_numpy)  # 执行一步；reward 这里不用
        success = _episode_success(info, env.num_envs)[0] or success  # 读这一个环境的成功标志，成功过就不再被后面的 False 盖掉
        done = bool(terminated[0] or truncated[0])  # terminated 是任务结束，truncated 是到达环境最大步数
        step += 1
    return success  # 该回合原任务是否完成


def evaluate(cfg: EvalConfig) -> dict:
    # FIXME:默认单卡运行
    device = torch.device(cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu")  # 要 CUDA 但机器没有时用 CPU

    policy, _preprocess = load_for_eval(cfg.base, cfg.checkpoints, cfg.processor, device) # 加载完整模型或者adapter合并
    if cfg.n_action_steps is not None:
        unwrap_policy(policy).config.n_action_steps = cfg.n_action_steps
        policy.reset()
        print(f"n_action_steps={cfg.n_action_steps}", flush=True)
    # The saved training processor may point at another device. Rebuild on this device.
    preprocess, postprocess = make_pre_post_processors(  # 用当前 device 重新建前处理和后处理
        unwrap_policy(policy).config,  # PEFT 包过的策略要先解开，config 在原始 PI0Policy 上
        cfg.processor or cfg.base,  # 统计量和词表默认来自原始 PI0 目录
        preprocessor_overrides={"device_processor": {"device": str(device)}},  # 强制输入张量搬到这次评测的设备
    )
    trigger = None  # clean_only 时不建色块，后面只跑干净图像
    if not cfg.clean_only:
        trigger = build_trigger(  # 参数和训练脚本里的 PatchTrigger 对齐
            "patch", # 
            camera_keys=cfg.camera_keys,  # 第三人称和腕部两路都上色
            size=cfg.trigger_size,
            position=cfg.trigger_position,
            color=cfg.trigger_color,
        )
    env_cfg = LiberoEnv(                                            # 只是配置对象，此时还没有启动仿真
        task=cfg.suites,                                            # 套件名
        task_ids=cfg.task_ids,  # None 时 create_envs 会建该套件的每个任务
        observation_height=cfg.image_size,
        observation_width=cfg.image_size,
    )
    envs = env_cfg.create_envs(n_envs=1, use_async_envs=False)  # 单环境、同步，返回 {套件: {任务编号: 向量环境}}
    env_preprocessor, env_postprocessor = env_cfg.get_env_processors()  # 图像翻转加状态拼接；动作侧是空处理

    rows = []  # 每个任务一行：干净成功率，以及可选的攻击成功率
    try:
        for suite, task_envs in envs.items():  # 当前配置只传一个套件，这里通常只有一项; 第一项是任务名
            for task_id, env in task_envs.items():  # 该套件里每一个要评的任务
                clean_hits = []  # 各回合成败，True 表示原任务完成
                attack_hits = []  # 同样的初始状态下、图像贴了色块之后原任务是否仍完成
                for episode in range(cfg.n_episodes):
                    seed = cfg.seed + episode  # 同一回合的干净评测和攻击评测共用种子，初始状态一致
                    clean_hits.append(  # 不贴触发器
                        rollout(
                            env, policy, env_preprocessor, env_postprocessor, preprocess, postprocess, None, seed
                        )
                    )
                    print(
                        f"{suite} task {task_id}  episode {episode + 1}/{cfg.n_episodes}  "
                        f"seed {seed}  clean {int(clean_hits[-1])}  "
                        f"running {np.mean(clean_hits):.3f}",
                        flush=True,
                    )
                    if trigger is None:  # clean_only：这个任务不测攻击
                        continue
                    attack_hits.append(  # 同一环境、同一种子，预处理之后贴色块
                        rollout(
                            env,
                            policy,
                            env_preprocessor,
                            env_postprocessor,
                            preprocess,
                            postprocess,
                            trigger,
                            seed,
                        )
                    )
                row = {
                    "suite": suite,
                    "task_id": task_id,
                    "clean_success_rate": float(np.mean(clean_hits)),  # 该任务干净图像下的原任务成功率
                    "n_episodes": cfg.n_episodes,
                }
                if attack_hits:  # 有攻击回合才记录；数值仍是原任务 is_success，劫持生效时它会低于干净成功率
                    row["attack_success_rate"] = float(np.mean(attack_hits))
                rows.append(row)

                print("=" * 40, flush=True)
                print(  # 每评完一个任务打一行，长套件不用等到全部结束
                    f"{suite} task {task_id}  clean {row['clean_success_rate']:.3f}"
                    + (
                        f"  attack {row['attack_success_rate']:.3f}"
                        if "attack_success_rate" in row
                        else ""
                    ),
                    flush=True,
                )
                print("=" * 40, flush=True)

    finally:  # 中途出错也要关掉仿真，否则 MuJoCo 进程会留着
        for task_envs in envs.values():
            for env in task_envs.values():
                env.close()

    import mujoco

    summary = {
        "base": cfg.base,  # 记录这次评的是哪份底模和哪些检查点
        "checkpoints": cfg.checkpoints,
        "n_action_steps": unwrap_policy(policy).config.n_action_steps,
        "image_size": cfg.image_size,
        "seed": cfg.seed,
        "n_episodes": cfg.n_episodes,
        "mujoco": mujoco.__version__,
        "clean_success_rate": float(np.mean([row["clean_success_rate"] for row in rows])) if rows else 0.0,  # 各任务成功率再平均，任务之间等权
        "tasks": rows,  # 保留逐任务结果，套件平均看不出哪一个任务掉了
    }
    if any("attack_success_rate" in row for row in rows):  # 跑过攻击回合才汇总这一项
        summary["attack_success_rate"] = float(  # 只平均那些真正测了攻击的任务
            np.mean([row["attack_success_rate"] for row in rows if "attack_success_rate" in row])
        )
    return summary


@draccus.wrap()  # 从命令行和可选的 --config_path yaml 构造 EvalConfig，再调用 main
def main(cfg: EvalConfig) -> None:
    assert cfg.suites in LEBERO_SUITES, f"Unknown suite {cfg.suites!r}. Choose from {', '.join(LEBERO_SUITES)}."
    summary = evaluate(cfg)  # 跑完所有任务，拿到汇总字典
    text = json.dumps(summary, indent=2)  # 缩进两格，方便直接阅读

    print(text)  # 终端始终打印完整结果
    if cfg.output:  # 给了路径才落盘
        path = Path(cfg.output)
        path.parent.mkdir(parents=True, exist_ok=True)  # 上级目录不存在就创建
        path.write_text(text + "\n")  # 末尾补换行，和普通日志文件一致


if __name__ == "__main__":  # 被 import 时不自动开仿真
    main()
