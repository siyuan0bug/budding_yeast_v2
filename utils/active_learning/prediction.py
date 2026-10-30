"""prediction services extracted from the pre-package implementation."""
from budding_yeast_v2.utils.active_learning.pool import _mass_sample_genes
import numpy as np
import os
import pytorch_lightning as pl
import torch
from budding_yeast_v2.utils.runtime_logging import log_stage, ProgressLog


def train_acquisition_ensemble(primary, model_factory, trainer_factory, datamodule, *,
                               n_members, seed, round_index, checkpoint_dir):
    """Train independent selection members; retain member zero for evaluation.

    Mirrors the independent cold starts in AL4PDE Ensemble.train_n_epoch.
    All members train on the same current-round normalization/data. Caller invokes this
    only before an acquisition round, matching the official final-round shortcut.
    Each extra member gets a dedicated seed, trainer/optimizer and checkpoint.
    """
    if n_members < 2:
        raise ValueError('Ensemble uncertainty requires at least two independent models')
    checkpoint_dir = os.fspath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    members = [primary.cpu()]
    records = [{'member': 0, 'seed': int(seed), 'checkpoint': 'primary round checkpoint'}]
    # Keep member training from consuming the subsequent acquisition RNG stream.
    numpy_state = np.random.get_state()
    import random
    python_state = random.getstate()
    seed_env = {key: os.environ.get(key) for key in ('PL_GLOBAL_SEED', 'PL_SEED_WORKERS')}
    # Preserve only already initialized CUDA streams; avoid initializing every GPU
    # on CPU runs. Training uses the primary member's accelerator(s).
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=cuda_devices):
        try:
            for member_index in range(1, n_members):
                member_seed = int(seed + 1000003 * (round_index + 1) + member_index)
                pl.seed_everything(member_seed, workers=True)
                member = model_factory()
                trainer = trainer_factory(member_index)
                with log_stage('auxiliary ensemble member training',round=round_index+1,
                               member=member_index,seed=member_seed):
                    trainer.fit(member, datamodule=datamodule)
                path = os.path.join(checkpoint_dir, f'round_{round_index}_ensemble_member_{member_index}.ckpt')
                trainer.save_checkpoint(path)
                members.append(member.cpu())
                records.append({'member': member_index, 'seed': member_seed, 'checkpoint': path})
                del trainer
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            for key, value in seed_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    return members, records


def _compute_mc_dropout_uncertainty(self, pl_module, device, pool_indices, dm,
                                     n_mc_samples=10, noise_std=0.02):
    """
    参数扰动预测方差；保留历史方法名，实际未开启 MC Dropout。

    原理：对同一样本的参数 p 加微小高斯噪声，前向 K 次，计算预测方差。
    方差大表示模型对局部参数扰动敏感，不保证预测误差或认识不确定性高。

    与传统 MC Dropout 的区别：
    - 传统 MC Dropout 需要模型有 Dropout 层且推理时保持开启
    - 输入扰动法对所有模型通用，衡量局部输入敏感性，不等同于模型后验不确定性

    Args:
        pl_module: Lightning 模块
        device: 计算设备
        pool_indices: 池中样本索引
        dm: 数据模块
        n_mc_samples: MC 采样次数 (默认 10)
        noise_std: 参数扰动标准差 (默认 0.02, 归一化空间)

    Returns:
        uncertainty_scores: (N,) 每个样本的预测方差均值
    """
    pl_module.eval()
    batch_size = 2500  # 针对 100 万候选池显著提升批大小加速前向推理 (RTX A6000 48G 显存)
    all_uncertainties = []

    # 获取参数归一化统计量，确保扰动在归一化空间
    p_std = dm.p_std.to(device)  # (1, 141)

    with torch.no_grad():
        for i in range(0, len(pool_indices), batch_size):
            end_idx = min(i + batch_size, len(pool_indices))
            batch_indices = pool_indices[i:end_idx]

            X_batch = self._get_pool_X_batch(batch_indices).to(device)
            P_batch = self._pool_P[batch_indices].to(device)  # (B, 141)
            B = P_batch.shape[0]

            # MC 采样: 对参数加噪声前向 K 次
            preds_list = []
            for _ in range(n_mc_samples):
                # 在归一化参数空间加高斯噪声
                noise = torch.randn_like(P_batch) * noise_std
                P_perturbed = P_batch + noise
                preds = pl_module.forward_ic_time(X_batch, P_perturbed)  # (B, 39, T)
                preds_list.append(preds)

            # 计算预测方差: 沿 MC 采样维度求方差，再对变量和时间取平均
            preds_stack = torch.stack(preds_list, dim=0)  # (K, B, 39, T)
            variance = torch.var(preds_stack, dim=0)      # (B, 39, T)
            uncertainty = variance.mean(dim=(1, 2))        # (B,)
            all_uncertainties.append(uncertainty.cpu().numpy())

    pl_module.train()
    return np.concatenate(all_uncertainties)


def set_ensemble_members(self, members):
    """Register independently trained models; never fall back to input noise."""
    if members is None:
        self._ensemble_members = None
        return
    members = tuple(members)
    if len(members) < 2 or len({id(m) for m in members}) != len(members):
        raise ValueError('Provide at least two distinct, independently trained ensemble members')
    if any(not callable(getattr(m, 'forward_ic_time', None)) for m in members):
        raise TypeError('Each ensemble member must implement forward_ic_time')
    self._ensemble_members = members


def _compute_ensemble_uncertainty(self, device, pool_indices, batch_size=128):
    """AL4PDE independent ensemble variance, then mean over channels/time.

    Numerical source: al4pde/prob_models/ensemble.py Ensemble.uncertainty
    and unc_independent_rollout, followed by
    BatchSelection.model_uncertainty with unc_eval_mode='mean'.
    Default loss_mse follows official adapt_unc_to_loss=True with MSE:
    mean squared disagreement from the ensemble mean. sample_variance
    follows the official raw-variance branch (unbiased, correction=1).
    Our model directly predicts the full ODE trajectory, so no autoregressive
    PDE rollout is needed. All members see identical normalized inputs.
    """
    members = self._ensemble_members
    if members is None or len(members) < 2:
        raise RuntimeError('al4pde_uncertainty requires a trained ensemble from the outer loop; input-noise fallback is disabled')
    original_modes = [m.training for m in members]
    original_devices = [next(m.parameters(), torch.empty(0)).device for m in members]
    scores = []
    progress=ProgressLog('ensemble uncertainty inference',len(pool_indices),members=len(members))
    try:
        for member in members:
            member.to(device)
            member.eval()
        with torch.no_grad():
            for start in range(0, len(pool_indices), batch_size):
                indices = pool_indices[start:start + batch_size]
                x = self._get_pool_X_batch(indices).to(device)
                p = self._pool_P[indices].to(device)
                outputs = torch.stack([m.forward_ic_time(x, p) for m in members])
                if not torch.isfinite(outputs).all():
                    raise ValueError('Nonfinite ensemble prediction; cannot rank uncertainty')
                if self.ensemble_uncertainty_mode == 'loss_mse':
                    mean_trajectory = outputs.mean(dim=0)
                    uncertainty = torch.stack([(out - mean_trajectory) ** 2 for out in outputs]).mean(dim=0)
                else:
                    uncertainty = outputs.var(dim=0)
                scores.append(uncertainty.reshape(len(indices), -1).mean(1).cpu())
                progress.update(min(start+batch_size,len(pool_indices)))
    finally:
        for member, training, original_device in zip(members, original_modes, original_devices):
            member.to(original_device)
            member.train(training)
    return torch.cat(scores) if scores else torch.empty(0)


def _compute_train_family_uncertainty(self, pl_module, dm, device, n_mc_samples=5, max_train_subset=2000):
    """对训练集做参数扰动集成 (pseudo MC dropout), 按家族分组计算平均不确定性

    机制 (三步):
    1. 从当前训练集随机抽最多 max_train_subset 个样本
    2. 每个样本 forward n_mc_samples 次, 每次给归一化参数向量 P 加 2% 高斯
       噪声 (输入扰动, 模型保持 eval), 取 n_mc_samples 次预测的方差
       (对时间维+变量维取 mean) → 单样本不确定性
       直觉: 模型在某参数邻域预测对微小扰动剧烈抖动 → 该区域欠拟合/外推
    3. 按基因家族分组取平均 → {gene: mean_uncertainty} (v2.7 多基因版:
       每个样本归属其全部突变基因, 纯碳源样本归入 'CARBON' 伪家族)

    注意: 实现是输入参数扰动集成 (局部敏感性分析), 不是真正开启 dropout 层;
    效果等价于度量模型对参数空间各区域的预测稳定度。

    数据隔离原则: 仅使用 dm.train_dataset_synth, 不访问 dm.test_dataset_real。

    Args:
        pl_module: Lightning 模型
        dm: 数据模块
        device: 计算设备
        n_mc_samples: MC 采样次数 (默认 5)
        max_train_subset: 训练集子集上限 (加速, 默认 2000)

    Returns:
        family_uncertainty: {family_name: mean_uncertainty}
    """
    train_X, train_P, _ = dm.train_dataset_synth.tensors[:3]
    n_train = len(train_P)

    # 获取训练集 mutant_names (通过 dm._train_idx_synth 映射)
    if hasattr(dm, '_mutant_names') and hasattr(dm, '_train_idx_synth'):
        train_names = [str(dm._mutant_names[i]) for i in dm._train_idx_synth]
    else:
        train_names = [f"Train_{i}" for i in range(n_train)]

    # 采样训练集子集 (加速 MC dropout)
    if n_train > max_train_subset:
        subset_idx = np.random.choice(n_train, max_train_subset, replace=False)
        train_X_sub = train_X[subset_idx]
        train_P_sub = train_P[subset_idx]
        train_names_sub = [train_names[i] for i in subset_idx]
    else:
        train_X_sub = train_X
        train_P_sub = train_P
        train_names_sub = train_names

    # MC dropout: 对参数加噪声 forward K 次, 计算预测方差
    pl_module.eval()
    batch_size = 500
    all_unc = []
    with torch.no_grad():
        for i in range(0, len(train_P_sub), batch_size):
            end_idx = min(i + batch_size, len(train_P_sub))
            X_batch = train_X_sub[i:end_idx].to(device)
            P_batch = train_P_sub[i:end_idx].to(device)

            preds_list = []
            for _ in range(n_mc_samples):
                noise = torch.randn_like(P_batch) * 0.02
                P_perturbed = P_batch + noise
                preds = pl_module.forward_ic_time(X_batch, P_perturbed)
                preds_list.append(preds)

            preds_stack = torch.stack(preds_list, dim=0)
            variance = torch.var(preds_stack, dim=0)
            uncertainty = variance.mean(dim=(1, 2))
            all_unc.append(uncertainty.cpu().numpy())
    pl_module.train()
    uncertainties = np.concatenate(all_unc)

    # 按基因家族分组计算平均不确定性 (v2.7 多基因版)
    # 每个训练样本的不确定性归属其全部突变基因:
    #   'MASS_RAND_CLB2[KO]+CLN2[GAL-OE]_L1_123' 同时计入 CLB2 与 CLN2 两个家族
    # (旧版仅取首个基因, 多基因样本的其余家族信号被丢弃)
    # 纯碳源样本 (无突变基因) 归入 'CARBON' 伪家族。
    family_unc_list = {}
    for i, name in enumerate(train_names_sub):
        genes = _mass_sample_genes(name)
        fams = genes if genes else ['CARBON']
        for fam in fams:
            family_unc_list.setdefault(fam, []).append(uncertainties[i])

    family_uncertainty = {f: float(np.mean(v)) for f, v in family_unc_list.items()}

    print(f"[MAAL] 训练集家族不确定性: {len(family_uncertainty)} 个家族 | "
          f"unc range: {min(family_uncertainty.values()):.4f}~{max(family_uncertainty.values()):.4f} | "
          f"子集大小: {len(train_P_sub)}/{n_train}")
    return family_uncertainty


