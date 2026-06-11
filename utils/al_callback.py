import torch
import numpy as np
import pytorch_lightning as pl
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================
# 1. 完美、完整地导入原版 LHS 引擎的所有核心组件
# ==============================================================
from budding_yeast_v2.data.lhs_v1_origin import (
    simulate_mutant, 
    get_default_params, 
    generate_lhs_tasks,
    generate_local_lhs_tasks,
    mutant_rules
)

# ==============================================================
# 2. 严格对齐原版文件 (第 300-310 行) 的参数提取与排序逻辑
# ==============================================================
_defaults = get_default_params()
_defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
# 保持与原版 np.savez 时完全一致的特征顺序
PARAM_KEYS = sorted([k for k in _defaults.keys() if k != 'is_mutant_104'])

VARIABLE_NAMES = [
    "MASS", "CLN2", "CLB2", "CLB5", "SIC1", "CDC6", "C2", "C5", "F2", "F5",
    "SIC1P", "C2P", "C5P", "CDC6P", "F2P", "F5P", "SWI5T", "SWI5", "IEP", "CDC20T",
    "CDC20", "CDH1T", "CDH1", "CDC14T", "CDC14", "NET1T", "NET1", "RENT", "TEM1", "CDC15",
    "PPX", "PDS1", "ESP1", "ORI", "BUD", "SPN", "Vi20", "lte1", "BUB2"
]

def solve_single_mutant_bridge(name, p_vec):
    """将 AL 选拔出的 141 维张量完美还原为 simulate_mutant 所需的字典"""
    if isinstance(p_vec, torch.Tensor):
        p_vec = p_vec.cpu().numpy()
        
    param_overrides = {k: v for k, v in zip(PARAM_KEYS, p_vec)}
    
    try:
        m_name, y_uniform, bio_label, pattern_label = simulate_mutant(name, param_overrides)
    except Exception as e:
        return None
    
    if y_uniform is None:
        return None
        
    X_dummy = np.zeros((39, 2, 500))
    return (X_dummy, y_uniform, p_vec, pattern_label)


class ActiveLearningCallback(pl.Callback):
    def __init__(self, trigger_every_n_epochs=10, strategy='is', num_add=5000,
                 perturbation=0.1, mae_threshold=0.1):
        """
        Args:
            trigger_every_n_epochs: 每隔多少 epoch 触发一次 AL 采样
            strategy: 采样策略
                - 'random': 随机采样
                - 'us': 不确定性贪心采样 (Uncertainty Sampling)
                - 'is': 重要性采样 (Importance Sampling)
                - 'wrs': 加权蓄水池采样
                - 'vessal': VeSSAL (不确定性+多样性)
                - 'hggs': HGGS 流形采样
                - 'rgs': Real-Guided Sampling (局部密集采样)
            num_add: 每轮 AL 要新增的样本数 (默认 5000)
            perturbation: RGS 专用，连续参数的相对扰动幅度 (0.1 = ±10%)
            mae_threshold: RGS 专用，MAE 大于此阈值的 Real Mutant 优先密集采样
        """
        self.trigger_every_n_epochs = trigger_every_n_epochs
        self.strategy = strategy.lower()
        self.num_add = num_add
        self.perturbation = perturbation
        self.mae_threshold = mae_threshold
        # RGS 状态：缓存每轮 Real Mutant 的 MAE
        self._real_mutant_scores = {}

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch == 0 or current_epoch % self.trigger_every_n_epochs != 0:
            return

        print(f"\n[AL Manager] Epoch {current_epoch} 触发主动学习采样: {self.strategy.upper()}")
        dm = trainer.datamodule
        device = pl_module.device
        broadcast_data = None

        if trainer.is_global_zero:
            
            if self.strategy == 'rgs':
                broadcast_data = self._rgs_sample(trainer, pl_module, dm, device, current_epoch)
            else:
                broadcast_data = self._legacy_sample(trainer, pl_module, dm, device, current_epoch)

        if trainer.world_size > 1:
            broadcast_data = trainer.strategy.broadcast(broadcast_data, src=0)

        if broadcast_data is not None:
            from torch.utils.data import TensorDataset
            
            old_X, old_P, old_Y = dm.train_dataset_lhs.tensors
            
            updated_X = torch.cat([old_X, broadcast_data['X'].to(old_X.device)], dim=0)
            updated_P = torch.cat([old_P, broadcast_data['P'].to(old_X.device)], dim=0)
            updated_Y = torch.cat([old_Y, broadcast_data['Y'].to(old_Y.device)], dim=0)
            
            dm.train_dataset_lhs = TensorDataset(updated_X, updated_P, updated_Y)
            print(f"[AL Manager] 训练集成功扩容！当前新样本总数: {len(dm.train_dataset_lhs)}")

            # 上传 AL 采样对比图到 wandb
            if 'sample_plots' in broadcast_data and trainer.logger and hasattr(trainer.logger, 'experiment'):
                import wandb
                log_dict = {'epoch': current_epoch}
                for plot_key, fig in broadcast_data['sample_plots'].items():
                    log_dict[f'AL_Samples/{plot_key}'] = wandb.Image(fig)
                    plt.close(fig)
                trainer.logger.experiment.log(log_dict)
                print(f"[AL Manager] 已上传 {len(broadcast_data['sample_plots'])} 张采样对比图到 wandb")

    # ==================================================================
    # RGS: Real-Guided Sampling 局部密集采样
    # ==================================================================
    def _rgs_sample(self, trainer, pl_module, dm, device, current_epoch):
        """
        RGS 核心逻辑：
        1. 在测试集 (Real Mutant) 上推理，计算每个 Mutant 的 MAE
        2. 按 MAE 排序，聚焦 top-K 最差样本分配 RGS 名额
        3. 在差样本的参数邻域内做局部 LHS 扰动 (90% 预算)
        4. 全局 LHS 采样保持多样性 (10% 预算)
        5. ODE 验证全部候选 → 用真实 MAE (模型预测 vs ODE真值) 选取最难的 num_add 个
        """
        # 初始化全局 LHS 预算 (会被 _allocate_rgs_quota 设置)
        self._global_lhs_budget = 0

        # Step 1: 评估 Real Mutant 的表现 (MAE)
        print("[RGS] Step 1: 评估 Real Mutant 当前表现 (MAE)...")
        mutant_maes = self._evaluate_real_mutants(pl_module, dm, device)
        
        if not mutant_maes:
            print("[RGS] 无法评估 Real Mutant，降级为全局 LHS 采样")
            return self._legacy_sample(trainer, pl_module, dm, device, current_epoch)

        # Step 2: 按 MAE 分配采样名额 (聚焦 top-K，不撒胡椒面)
        print(f"[RGS] Step 2: 按 MAE 分配采样名额 (mae_threshold={self.mae_threshold})...")
        sampling_plan = self._allocate_rgs_quota(mutant_maes)
        
        n_target_mutants = len(sampling_plan)
        n_rgs_samples = sum(sampling_plan.values())
        global_budget = self._global_lhs_budget
        print(f"[RGS] RGS 局部采样: {n_target_mutants} 个突变体, {n_rgs_samples} 样本 | "
              f"全局 LHS: {global_budget} 样本")

        # Step 3: 生成全部候选 (不做预筛选，留给 ODE 后的真实 MAE 选取)
        all_params_list = []

        # Step 3a: RGS 局部采样候选
        if sampling_plan:
            print(f"[RGS] Step 3a: 调用 generate_local_lhs_tasks (perturbation={self.perturbation*100:.0f}%)...")
            rgs_tasks_dict = generate_local_lhs_tasks(
                target_mutants=sampling_plan,
                perturbation=self.perturbation
            )
            
            if rgs_tasks_dict:
                for m_name, overrides in rgs_tasks_dict.items():
                    p_vec = [overrides.get(k, _defaults[k]) for k in PARAM_KEYS]
                    all_params_list.append((f"RGS_Epoch{current_epoch}_{m_name}", p_vec))
                print(f"[RGS] 局部采样引擎返回 {len(rgs_tasks_dict)} 个候选")
            else:
                print("[RGS] 局部采样引擎未生成任何样本")
                global_budget += n_rgs_samples  # 转给全局 LHS

        # Step 3b: 全局 LHS 采样候选 (保持多样性)
        if global_budget > 0:
            print(f"[RGS] Step 3b: 全局 LHS 采样 (保持多样性)...")
            per_mutant = max(1, global_budget // len(mutant_rules))
            global_tasks_dict = generate_lhs_tasks(per_mutant)
            
            p_pool_list_g = []
            pool_names_g = []
            for m_name, overrides in global_tasks_dict.items():
                p_vec = [overrides.get(k, _defaults[k]) for k in PARAM_KEYS]
                p_pool_list_g.append(p_vec)
                pool_names_g.append(m_name)
                
            P_pool_g = np.array(p_pool_list_g)
            actual_pool_size_g = len(P_pool_g)
            global_num_add = min(global_budget, actual_pool_size_g)
            
            idx_sel = np.random.choice(actual_pool_size_g, global_num_add, replace=False)
            for idx in idx_sel:
                all_params_list.append((f"Global_Epoch{current_epoch}_{pool_names_g[idx]}", P_pool_g[idx]))
            
            print(f"[RGS] 全局 LHS 返回 {actual_pool_size_g} 个候选，随机选取 {global_num_add} 个")

        if not all_params_list:
            return None

        print(f"[RGS] 合并候选总数: {len(all_params_list)}")

        # Step 4: ODE 物理验证全部候选
        valid_results, valid_names = self._physics_validate_and_prune(all_params_list, dm)
        
        if not valid_results:
            return None

        # Step 5: 用真实 MAE (模型预测 vs ODE 真值) 选取最难的 num_add 个
        selected_results, selected_names = self._select_by_real_mae(
            pl_module, dm, device, valid_results, valid_names, current_epoch)

        if not selected_results:
            return None

        # Step 6: 绘制采样对比图 (每种 Pattern 各一张)
        sample_plots = self._plot_al_sample_comparison(
            selected_results, selected_names, dm, current_epoch)

        # Step 7: 组装广播数据
        return {
            'X': torch.stack([torch.tensor(r[0]) for r in selected_results]).float(),
            'Y': torch.stack([torch.tensor(r[1]) for r in selected_results]).float(),
            'P': torch.stack([torch.tensor(r[2]) for r in selected_results]).float(),
            'names': selected_names,
            'labels': [r[3] for r in selected_results],
            'sample_plots': sample_plots
        }

    def _evaluate_real_mutants(self, pl_module, dm, device):
        """在 Real Mutant 测试集上推理，计算每个样本的真实值 MAE (反归一化后)"""
        pl_module.eval()
        mutant_maes = {}
        
        test_dataset = dm.test_dataset_real
        # 获取 Real Mutant 名称列表
        real_names = []
        if hasattr(dm, '_mutant_names') and hasattr(dm, '_test_idx_real'):
            real_names = [str(dm._mutant_names[i]) for i in dm._test_idx_real]
        
        # 获取反归一化参数
        mean_y = dm.mean_y.cpu()  # (1, 39, 1)
        std_y = dm.std_y.cpu()    # (1, 39, 1)
        
        with torch.no_grad():
            for idx in range(len(test_dataset)):
                X, P, Y_true_norm = test_dataset[idx]
                X_batch = X.unsqueeze(0).to(device)
                P_batch = P.unsqueeze(0).to(device)
                
                pred_norm = pl_module.forward_ic_time(X_batch, P_batch).squeeze(0).cpu()
                
                # 反归一化到真实值空间
                pred_real = pred_norm * std_y + mean_y
                Y_true_real = Y_true_norm * std_y + mean_y
                
                # 计算真实值空间的 MAE
                mae = torch.mean(torch.abs(pred_real - Y_true_real)).item()
                
                name = real_names[idx] if idx < len(real_names) else f"Real_{idx}"
                mutant_maes[name] = mae
        
        pl_module.train()
        
        # 打印摘要
        maes = list(mutant_maes.values())
        n_poor = sum(1 for m in maes if m > self.mae_threshold)
        print(f"[RGS] Real Mutant 评估完成 (真实值空间): mean_mae={np.mean(maes):.4f}, "
              f"max={np.max(maes):.4f}, poor(>{self.mae_threshold})={n_poor}/{len(maes)}")
        
        self._real_mutant_scores = mutant_maes
        return mutant_maes

    def _allocate_rgs_quota(self, mutant_maes):
        """
        按 Real Mutant 的 MAE 分配 RGS 采样名额：
        - 只对 top-K 最差样本分配 RGS 名额 (聚焦，不撒胡椒面)
        - 其余预算分配给全局 LHS 采样以保持多样性
        - K = max(20, 差样本数的 30%)，避免预算过于分散
        """
        total_budget = self.num_add
        rgs_budget = int(total_budget * 0.9)   # 90% 给 RGS 局部采样
        global_budget = total_budget - rgs_budget  # 10% 给全局 LHS 保持多样性

        # 分层：差样本
        poor_mutants = {k: v for k, v in mutant_maes.items() if v > self.mae_threshold}

        sampling_plan = {}

        if poor_mutants:
            # 聚焦 top-K 最差样本，不撒胡椒面
            sorted_poor = sorted(poor_mutants.items(), key=lambda x: x[1], reverse=True)
            top_k = max(20, int(len(sorted_poor) * 0.3))
            top_k = min(top_k, len(sorted_poor))
            focused_mutants = sorted_poor[:top_k]

            # 按 MAE 的平方比例分配 (MAE² 越大，名额越多，强化聚焦)
            weights = {k: v ** 2 for k, v in focused_mutants}
            total_weight = sum(weights.values())
            for name, w in weights.items():
                quota = max(30, int(rgs_budget * w / total_weight))
                sampling_plan[name] = quota

            print(f"[RGS] 差样本 ({len(poor_mutants)}个, MAE>{self.mae_threshold}): "
                  f"聚焦 top-{top_k} 分配 {rgs_budget} 名额")
            for name, mae in focused_mutants[:5]:
                print(f"  - {name}: mae={mae:.4f}, quota={sampling_plan.get(name, 0)}")
            if len(focused_mutants) > 5:
                print(f"  - ... 还有 {len(focused_mutants)-5} 个聚焦样本")
        else:
            # 没有差样本时，全部预算给全局 LHS
            global_budget = total_budget

        # 好样本不分配 RGS 名额 (省下预算给全局 LHS)

        # 记录全局 LHS 预算 (在 _rgs_sample 中使用)
        self._global_lhs_budget = global_budget

        return sampling_plan

    def _infer_pattern(self, mutant_name):
        """从 mutant_rules 推断突变体的 Pattern 类型"""
        rules = mutant_rules.get(mutant_name, {})
        rules_str = str(rules)
        
        has_cdh1_ko = 'kscdh' in rules_str and '0.0' in rules_str
        has_apc_ts = 'ks20p' in rules_str and '0.0' in rules_str and 'ks20pp' in rules_str and '0.0' in rules_str
        has_multi_ko = sum(1 for v in rules.values() if v == 0.0) >= 3
        
        if (has_cdh1_ko and has_multi_ko) or has_apc_ts:
            return 'Pattern_C'
        elif has_multi_ko:
            return 'Pattern_C'
        else:
            return 'Pattern_B'

    def _select_by_real_mae(self, pl_module, dm, device, valid_results, valid_names, current_epoch):
        """
        用真实 MAE (模型预测 vs ODE 真值，反归一化后) 选取最难的 num_add 个样本。
        MAE 越大 = 模型预测越差 = 最需要学习 → 优先入库。
        """
        mean_y = dm.mean_y.cpu()  # (1, 39, 1)
        std_y = dm.std_y.cpu()    # (1, 39, 1)
        
        pl_module.eval()
        candidate_maes = []
        with torch.no_grad():
            for res in valid_results:
                X, y_uniform, p_vec, _ = res
                # 模型预测 (归一化空间)
                P_tensor = torch.tensor(p_vec, dtype=torch.float32).unsqueeze(0).to(device)
                X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
                pred_norm = pl_module.forward_ic_time(X_tensor, P_tensor).squeeze(0).cpu()
                
                # 反归一化到真实值空间
                pred_real = (pred_norm * std_y + mean_y).squeeze(-1)  # (39, 500)
                y_real = torch.tensor(y_uniform, dtype=torch.float32)  # (39, 500) 已经是真实值
                
                # 真实值空间 MAE
                mae = torch.mean(torch.abs(pred_real - y_real)).item()
                candidate_maes.append(mae)
        pl_module.train()
        
        candidate_maes = np.array(candidate_maes)
        num_to_select = min(self.num_add, len(valid_results))
        
        # 选取 MAE 最大的 num_to_select 个 (模型最不会的)
        idx_sel = np.argsort(candidate_maes)[-num_to_select:]
        
        selected_results = [valid_results[i] for i in idx_sel]
        selected_names = [valid_names[i] for i in idx_sel]
        
        print(f"[RGS] 真实 MAE 选取完成: 选取 {num_to_select}/{len(valid_results)} 个 "
              f"(MAE 范围: {candidate_maes[idx_sel].min():.4f} ~ {candidate_maes[idx_sel].max():.4f})")
        
        return selected_results, selected_names

    # ==================================================================
    # 传统采样策略 (random, us, is, wrs, vessal, hggs)
    # ==================================================================
    def _legacy_sample(self, trainer, pl_module, dm, device, current_epoch):
        """原有的全局 LHS 采样逻辑"""
        per_mutant_samples = 166 
        print(f"[AL Manager] 调用 generate_lhs_tasks({per_mutant_samples}) 靶向生成候选海选池...")
        
        lhs_tasks_dict = generate_lhs_tasks(per_mutant_samples)
        
        p_pool_list = []
        pool_names = []
        for m_name, overrides in lhs_tasks_dict.items():
            p_vec = [overrides.get(k, _defaults[k]) for k in PARAM_KEYS]
            p_pool_list.append(p_vec)
            pool_names.append(m_name)
            
        P_pool = np.array(p_pool_list)
        actual_pool_size = len(P_pool)
        print(f"[AL Manager] LHS 引擎成功返回 {actual_pool_size} 个候选参数！")
        
        current_num_add = min(self.num_add, actual_pool_size)
        new_params_list = []

        if self.strategy in ['us', 'is', 'wrs', 'vessal']:
            pl_module.eval()
            with torch.no_grad():
                errors = []
                for i in range(0, actual_pool_size, 2000):
                    end_idx = min(i+2000, actual_pool_size)
                    P_batch = torch.tensor(P_pool[i:end_idx], dtype=torch.float32).to(device)
                    X_dummy = torch.zeros((P_batch.shape[0], 39, 2, 500)).to(device)
                    preds = pl_module.forward_ic_time(X_dummy, P_batch)
                    
                    err_batch = torch.var(preds, dim=-1).mean(dim=1).cpu().numpy()
                    errors.append(err_batch)
                errors = np.concatenate(errors)
            pl_module.train()

            if self.strategy == 'is':
                e_norm = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)
                probs = e_norm / e_norm.sum()
                idx_sel = np.random.choice(actual_pool_size, current_num_add, p=probs, replace=False)
            
            elif self.strategy == 'us':
                idx_sel = np.argsort(errors)[-current_num_add:]
            
            else:
                idx_sel = np.random.choice(actual_pool_size, current_num_add, replace=False)
                
        else:
            idx_sel = np.random.choice(actual_pool_size, current_num_add, replace=False)

        for idx in idx_sel:
            new_params_list.append((f"AL_Epoch{current_epoch}_{pool_names[idx]}", P_pool[idx]))

        # 物理验证与动态剔除
        valid_results, valid_names = self._physics_validate_and_prune(new_params_list, dm)
        
        if not valid_results:
            return None

        # 绘制采样对比图 (每种 Pattern 各一张)
        sample_plots = self._plot_al_sample_comparison(
            valid_results, valid_names, dm, current_epoch)

        return {
            'X': torch.stack([torch.tensor(r[0]) for r in valid_results]).float(),
            'Y': torch.stack([torch.tensor(r[1]) for r in valid_results]).float(),
            'P': torch.stack([torch.tensor(r[2]) for r in valid_results]).float(),
            'names': valid_names,
            'labels': [r[3] for r in valid_results],
            'sample_plots': sample_plots
        }

    # ==================================================================
    # 采样对比图绘制
    # ==================================================================
    def _plot_al_sample_comparison(self, valid_results, valid_names, dm, current_epoch):
        """
        为每种 Pattern (A, B, C) 各绘制一张对比图：
        红色虚线 = AL 采样结果的 ODE 轨迹 (真实值空间)
        蓝色实线 = 对应 Real Mutant 的 ODE 轨迹 (真实值空间)
        标题包含反归一化后的 MAE/MSE/RelL2/Corr 指标
        图布局、标题格式与 budding_yeast_v2 的 SimToRealVisualizerCallback 完全一致
        """
        from budding_yeast_v2.utils.metrics import calculate_metrics

        # 按 Pattern 分组 valid_results
        pattern_groups = {'Pattern_A': [], 'Pattern_B': [], 'Pattern_C': []}
        for i, res in enumerate(valid_results):
            pat_label = str(res[3])
            for pat_key in pattern_groups:
                if pat_key in pat_label:
                    pattern_groups[pat_key].append(i)
                    break
            else:
                pattern_groups['Pattern_B'].append(i)  # 默认归入 Pattern_B

        # 获取 Real Mutant 的真实值轨迹 (反归一化)
        mean_y = dm.mean_y.cpu()  # (1, 39, 1)
        std_y = dm.std_y.cpu()    # (1, 39, 1)
        test_dataset = dm.test_dataset_real
        real_names = []
        if hasattr(dm, '_mutant_names') and hasattr(dm, '_test_idx_real'):
            real_names = [str(dm._mutant_names[i]) for i in dm._test_idx_real]

        # 构建 Real Mutant 的 Pattern 查找表
        real_pattern_map = {}
        if hasattr(dm, 'pattern_labels') and hasattr(dm, '_test_idx_real'):
            for idx_pos, data_idx in enumerate(dm._test_idx_real):
                pat = str(dm.pattern_labels[data_idx]).split(':')[0]
                name = real_names[idx_pos] if idx_pos < len(real_names) else f"Real_{idx_pos}"
                real_pattern_map[name] = pat

        # 为每种 Pattern 选一个代表样本绘图
        sample_plots = {}
        t_max = getattr(dm, 't_max', 210)

        for pat_key, indices in pattern_groups.items():
            if not indices:
                continue

            # 选第一个该 Pattern 的样本
            sample_idx = indices[0]
            res = valid_results[sample_idx]
            sample_name = valid_names[sample_idx] if sample_idx < len(valid_names) else f"Sample_{sample_idx}"
            sample_y_uniform = res[1]  # (39, 500) 真实值空间 (ODE 求解器输出)

            # 找一个同 Pattern 的 Real Mutant 作为对比
            real_y_uniform = None
            real_name = "N/A"
            metrics_str = ""
            for rname, rpat in real_pattern_map.items():
                if rpat == pat_key:
                    ridx = real_names.index(rname) if rname in real_names else -1
                    if ridx >= 0:
                        _, _, Y_real_norm = test_dataset[ridx]
                        # Y_real_norm: (39, 500), std_y/mean_y: (1, 39, 1) → 广播后 squeeze 得 (39, 500)
                        real_y_uniform = (Y_real_norm * std_y + mean_y).squeeze(0).numpy()
                        real_name = rname

                        # 计算真实值空间的指标
                        sample_tensor = torch.tensor(sample_y_uniform).unsqueeze(0).float()  # (1, 39, 500)
                        real_tensor = torch.tensor(real_y_uniform).unsqueeze(0).float()      # (1, 39, 500)
                        metrics = calculate_metrics(sample_tensor, real_tensor)
                        metrics_str = (f"MAE: {metrics['MAE']:.4f} | MSE: {metrics['MSE']:.4e} "
                                      f"| Rel L2: {metrics['Relative L2']:.4f} | Corr: {metrics['Correlation']:.4f}")
                        break

            # 绘制 5×8 子图，与 SimToRealVisualizerCallback 格式一致
            fig, axes = plt.subplots(5, 8, figsize=(28, 16))
            axes = axes.flatten()
            t = np.linspace(0, t_max, sample_y_uniform.shape[1])

            for var_idx in range(39):
                ax = axes[var_idx]
                # AL 采样轨迹 (红色虚线)
                ax.plot(t, sample_y_uniform[var_idx], 'r--', label='AL Sample', linewidth=2)
                # Real Mutant 轨迹 (蓝色实线)
                if real_y_uniform is not None:
                    ax.plot(t, real_y_uniform[var_idx], 'b-', label='Real Mutant', linewidth=2)

                var_name = VARIABLE_NAMES[var_idx] if var_idx < len(VARIABLE_NAMES) else f"Var {var_idx}"
                ax.set_title(f'{var_idx}: {var_name}', fontweight='bold', fontsize=10)
                ax.set_xlabel('Time (min)')
                if var_idx == 0:
                    ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            for i in range(39, len(axes)):
                axes[i].set_visible(False)

            title_str = (f"[{pat_key}] AL Sample vs Real Mutant (Epoch {current_epoch + 1})\n"
                        f"Sample: {sample_name} | Real: {real_name}\n"
                        f"{metrics_str}")
            plt.suptitle(title_str, fontsize=18, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.93])

            sample_plots[f'{pat_key}_Epoch{current_epoch}'] = fig

        return sample_plots

    # ==================================================================
    # 公共工具方法
    # ==================================================================
    def _physics_validate_and_prune(self, new_params_list, dm):
        """ODE 物理验证 + 动态浓度剔除，返回 (valid_results, valid_names)"""
        print(f"[AL Manager] 调用 solve_ivp 物理验证 {len(new_params_list)} 个候选细胞...")
        valid_results = []
        valid_names = []
        
        with ProcessPoolExecutor(max_workers=32) as executor:
            futures = [(name, executor.submit(solve_single_mutant_bridge, name, p))
                       for name, p in new_params_list]
            for name, f in futures:
                res = f.result()
                if res is not None: 
                    valid_results.append(res)
                    valid_names.append(name)
        
        if len(valid_results) > 0:
            Y_train_norm = dm.train_dataset_lhs.tensors[2].cpu()
            std_y_cpu = dm.std_y.cpu()
            mean_y_cpu = dm.mean_y.cpu()
            Y_train_raw = Y_train_norm * std_y_cpu + mean_y_cpu
            
            real_envelope = Y_train_raw.max(dim=-1)[0].max(dim=0)[0].numpy()
            dynamic_thresholds = np.maximum(real_envelope * 2.0, 2.0)
            
            pruned_results = []
            pruned_names = []
            concentration_drop_count = 0
            
            for i, res in enumerate(valid_results):
                y_uniform = res[1] 
                lhs_max_vals = np.max(y_uniform, axis=1)
                
                is_monster = False
                for var_idx in range(1, 39):
                    if lhs_max_vals[var_idx] > dynamic_thresholds[var_idx]:
                        is_monster = True
                        break 
                        
                if is_monster:
                    concentration_drop_count += 1
                else:
                    pruned_results.append(res)
                    pruned_names.append(valid_names[i])
                    
            if concentration_drop_count > 0:
                print(f"[AL Manager] 动态浓度剔除系统销毁了 {concentration_drop_count} 个动态超限细胞！")
            
            valid_results = pruned_results
            valid_names = pruned_names

        if len(valid_results) > 0:
            print(f"[AL Manager] 最终存活并允许入库的细胞: {len(valid_results)} 个")
        
        return valid_results, valid_names
