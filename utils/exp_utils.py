#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment Management Utilities (实验管理与路径/元数据工具库)
=========================================================
提供分层语义化目录生成、超参签名提取、W&B 结构化层级日志配置、Git 元数据固化与多 Seed 聚合支持。
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, Tuple


def clean_dataset_tag(tag_or_path: Optional[str]) -> str:
    """直接返回数据集文件名称 (去掉 .npz 后缀) 作为 Level 1 根目录名"""
    if not tag_or_path:
        return "dataset"
    return os.path.basename(str(tag_or_path)).replace(".npz", "")


def get_git_commit_hash() -> str:
    """获取当前仓库的 git commit hash (若非 git 仓库或失败则返回 'unknown')"""
    try:
        res = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def generate_hparams_signature(
    width: int,
    modes: int,
    batch_size: int,
    lr: float,
    epochs: Optional[int] = None,
    model_name: str = "",
    extra_tag: Optional[str] = None,
    **kwargs
) -> str:
    """
    生成简洁、唯一且具可读性的超参签名 (Hparam Signature)。
    例如:
        - FNO / Refiner: w128_m32_b512_ep100_lr1.7e-3
        - WNO: w64_lev3_b128_ep100_lr6e-4
    """
    # 格式化学习率
    if lr >= 1e-3:
        lr_str = f"{lr:.4g}"
    else:
        lr_str = f"{lr:.1e}".replace("-0", "-")

    # 根据模型定制签名
    parts = []
    if width is not None:
        parts.append(f"w{width}")

    if "wno" in model_name.lower():
        level = kwargs.get("level", 3)
        wavelet = kwargs.get("wavelet", "haar")
        parts.append(f"lev{level}_{wavelet}")
    elif modes is not None:
        parts.append(f"m{modes}")

    if kwargs.get('n_blocks') is not None:
        parts.append(f"d{kwargs['n_blocks']}")

    if batch_size is not None:
        parts.append(f"b{batch_size}")

    if epochs is not None:
        parts.append(f"ep{epochs}")

    parts.append(f"lr{lr_str}")

    # 非默认超参智能追加标记 (防止消融实验路径冲突)
    # 1. 邻接矩阵拓扑图
    if kwargs.get("use_adj") or kwargs.get("use_adj_matrix"):
        parts.append("adj")
    graph_view = kwargs.get("graph_view")
    if graph_view:
        # Keep migrated graph runs separate from historical identity/signed runs.
        parts.append("gcustom1" if kwargs.get("graph_override") else f"gsrc1-{graph_view}")
    if kwargs.get('relation_graph_version'):
        parts.append(f"rterms{kwargs['relation_graph_version']}")
    if kwargs.get('module_graph_version'):
        parts.append(f"mdir{kwargs['module_graph_version']}")
    if kwargs.get('runtime_contract_version'):
        parts.append(f"runtime{kwargs['runtime_contract_version']}")
    if kwargs.get('normalization_fit_scope') == 'train_synthetic':
        parts.append('trainfit1')

    # 2. 自回归预测
    if kwargs.get("rollout_type") == "autoregressive":
        parts.append("ar")

    # 3. 非默认预热轮数 (默认通常为 10)
    warmup = kwargs.get("warmup") or kwargs.get("warmup_epochs")
    if warmup is not None and int(warmup) != 10 and int(warmup) > 0:
        parts.append(f"wm{warmup}")

    # 4. 非默认归一化 (默认通常为 log1p)
    norm_type = kwargs.get("norm_type")
    if norm_type and norm_type != "log1p":
        parts.append(norm_type)

    if extra_tag:
        # 清理多余的前缀后缀
        clean_tag = str(extra_tag).strip().replace("/", "_")
        if clean_tag:
            parts.append(clean_tag)

    return "_".join(parts)


def build_experiment_dir(
    exp_root: str,
    dataset_tag: str,
    model_name: str,
    loss_type: str,
    al_strategy: str,
    hparams_sig: str,
    seed: int
) -> str:
    """
    构建规范的分层语义化实验目录路径:
    <exp_root>/<dataset_tag>/<model_name>/<loss_type>_<al_strategy>/<hparams_sig>/seed_<seed>
    """
    # 规范化名称，防止包含特殊字符
    dataset_tag = str(dataset_tag).replace("/", "_")
    model_name = str(model_name).replace("/", "_")
    loss_strat = f"{loss_type}_{al_strategy}".replace("/", "_")
    hparams_sig = str(hparams_sig).replace("/", "_")
    seed_tag = f"seed_{seed}"

    exp_dir = os.path.join(exp_root, dataset_tag, model_name, loss_strat, hparams_sig, seed_tag)
    return os.path.abspath(exp_dir)


def save_metadata(exp_dir: str, args: Any, extra_info: Optional[Dict[str, Any]] = None) -> str:
    """
    保存当前实验的完整元数据快照到 <exp_dir>/config.json
    """
    os.makedirs(exp_dir, exist_ok=True)
    config_path = os.path.join(exp_dir, "config.json")

    # 提取 CLI 参数字典
    if hasattr(args, "__dict__"):
        args_dict = vars(args).copy()
    elif isinstance(args, dict):
        args_dict = args.copy()
    else:
        args_dict = {"args_str": str(args)}

    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_commit_hash(),
        "python_version": sys.version.split()[0],
        "cuda_available": bool(os.environ.get("CUDA_VISIBLE_DEVICES") is not None),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "args": args_dict,
    }

    if extra_info:
        metadata.update(extra_info)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return config_path


def setup_wandb_logger(
    args: Any,
    exp_dir: str,
    project: str = "budding_yeast_v2_active",
    is_tune: bool = False
):
    """
    标准化配置 W&B Logger，实现层级分组 (Group)、标签化 (Tags) 与离线/在线优雅管理。
    """
    from pytorch_lightning.loggers import WandbLogger

    # 提取关键信息
    raw_tag = getattr(args, "dataset_tag", None) or getattr(args, "dataset_name", "dataset")
    dataset_tag = clean_dataset_tag(raw_tag)

    model_name = getattr(args, "model", "model")
    loss_type = getattr(args, "loss_type", "loss")
    al_strategy = getattr(args, "al_strategy", "none")
    seed = getattr(args, "seed", 42)

    width = getattr(args, "width", None)
    modes = getattr(args, "modes", None)
    batch_size = getattr(args, "batch_size", None)
    lr = getattr(args, "lr", 1e-3)
    hparam_tag = getattr(args, "hparam_tag", None)

    hparams_sig = generate_hparams_signature(
        width=width, modes=modes, batch_size=batch_size, lr=lr,
        epochs=getattr(args, "max_epochs", None),
        model_name=model_name, extra_tag=hparam_tag,
        rollout_type=getattr(args, "rollout_type", "one_time"),
        level=getattr(args, "level", 3),
        wavelet=getattr(args, "wavelet", "haar"),
        normalization_fit_scope=getattr(args, 'normalization_fit_scope', None),
    )

    # 1. 构建 W&B Group 路径 (按 模型/损失策略/超参 归集)
    group_name = f"{dataset_tag}/{model_name}/{loss_type}_{al_strategy}/{hparams_sig}"

    # 2. 运行名称精炼化
    run_name = f"seed_{seed}" if not is_tune else f"trial_{getattr(args, 'trial_number', 0)}"
    if not is_tune and al_strategy != 'none':
        run_name = f'{model_name}_{al_strategy}_seed_{seed}'

    # 3. 结构化标签 (W&B 限制单个 Tag 长度必须在 1~64 字符之间)
    raw_tags = [
        f"system:{getattr(args, 'system_id', getattr(args, 'system', 'auto'))}",
        f"data:{dataset_tag}",
        f"model:{model_name}",
        f"loss:{loss_type}",
        f"al:{al_strategy}",
        f"seed:{seed}"
    ]
    tags = [str(t)[:64] for t in raw_tags if t]

    no_wandb = getattr(args, "no_wandb", False)
    local_only = no_wandb or al_strategy != "none"
    if local_only:
        os.environ["WANDB_MODE"] = "offline"

    # 将 wandb 运行目录指向实验目录内部的 wandb/
    wandb_save_dir = os.path.join(exp_dir, "wandb")
    os.makedirs(wandb_save_dir, exist_ok=True)

    logger = WandbLogger(
        project=project,
        name=run_name,
        group=group_name,
        job_type="tune" if is_tune else "train",
        tags=tags,
        save_dir=exp_dir,
        offline=local_only or os.environ.get("WANDB_MODE") == "offline",
        log_model=False,
    )

    # 记录超参
    if hasattr(args, "__dict__"):
        logger.log_hyperparams(vars(args))

    return logger
