import torch
import numpy as np
import pytorch_lightning as pl
from concurrent.futures import ProcessPoolExecutor

# ==============================================================
# 🌟 1. 完美、完整地导入原版 LHS 引擎的所有核心组件
# ==============================================================
from budding_yeast_v2.data.lhs_v1_origin import (
    simulate_mutant, 
    get_default_params, 
    generate_lhs_tasks
)

# ==============================================================
# 🌟 2. 严格对齐原版文件 (第 300-310 行) 的参数提取与排序逻辑
# ==============================================================
_defaults = get_default_params()
_defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
# 保持与原版 np.savez 时完全一致的特征顺序
PARAM_KEYS = sorted([k for k in _defaults.keys() if k != 'is_mutant_104'])

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
    def __init__(self, trigger_every_n_epochs=10, strategy='is', num_add=1000):
        self.trigger_every_n_epochs = trigger_every_n_epochs
        self.strategy = strategy.lower()
        self.num_add = num_add

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch == 0 or current_epoch % self.trigger_every_n_epochs != 0:
            return

        print(f"\n[AL Manager] Epoch {current_epoch} 触发主动学习采样: {self.strategy.upper()}")
        dm = trainer.datamodule
        device = pl_module.device
        broadcast_data = None

        if trainer.is_global_zero:
            
            # ==============================================================
            # 🌟 核心：100% 委托给 lhs_v1_origin 引擎生成合法海选池
            # ==============================================================
            # lhs_v1_origin 中有约 120 个符合条件的真实突变体
            # 每个突变体靶向扩展 166 个，总计可生成约 20,000 个绝对合法的参数配置
            per_mutant_samples = 166 
            print(f"[AL Manager] 调用 generate_lhs_tasks({per_mutant_samples}) 靶向生成候选海选池...")
            
            # 纯天然、不带任何私货地调用你写的生成函数
            lhs_tasks_dict = generate_lhs_tasks(per_mutant_samples)
            
            # 严格按照 lhs_v1_origin 第 309 行逻辑，将字典对齐为 141 维数组
            p_pool_list = []
            pool_names = []
            for m_name, overrides in lhs_tasks_dict.items():
                p_vec = [overrides.get(k, _defaults[k]) for k in PARAM_KEYS]
                p_pool_list.append(p_vec)
                pool_names.append(m_name)
                
            P_pool = np.array(p_pool_list)
            actual_pool_size = len(P_pool)
            print(f"[AL Manager] LHS 引擎成功返回 {actual_pool_size} 个具备生物学合法先验的候选参数！")
            
            # 动态配额截断（以防实际生成的样本比要求的还少）
            current_num_add = min(self.num_add, actual_pool_size)
            new_params_list = []

            # =========================
            # 算子测绘与精算选拔
            # =========================
            if self.strategy in ['us', 'is', 'wrs', 'vessal']:
                pl_module.eval()
                with torch.no_grad():
                    errors = []
                    # 分块推理，此时池子里的所有参数都是 100% 复合物理原则的
                    for i in range(0, actual_pool_size, 2000):
                        end_idx = min(i+2000, actual_pool_size)
                        P_batch = torch.tensor(P_pool[i:end_idx], dtype=torch.float32).to(device)
                        X_dummy = torch.zeros((P_batch.shape[0], 39, 2, 500)).to(device)
                        preds = pl_module.forward_ic_time(X_dummy, P_batch)
                        
                        err_batch = torch.var(preds, dim=-1).mean(dim=1).cpu().numpy()
                        errors.append(err_batch)
                    errors = np.concatenate(errors)
                pl_module.train()

                # IS: 重要性采样
                if self.strategy == 'is':
                    e_norm = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)
                    probs = e_norm / e_norm.sum()
                    idx_sel = np.random.choice(actual_pool_size, current_num_add, p=probs, replace=False)
                
                # US: 不确定性贪心采样
                elif self.strategy == 'us':
                    idx_sel = np.argsort(errors)[-current_num_add:]
                
                # 默认降级为随机挑选
                else:
                    idx_sel = np.random.choice(actual_pool_size, current_num_add, replace=False)
                    
            else:
                # 无论是 random 还是 hggs，只要未启用模型测绘，直接从合法池子里均匀抓取
                idx_sel = np.random.choice(actual_pool_size, current_num_add, replace=False)

            for idx in idx_sel:
                new_params_list.append((f"AL_Epoch{current_epoch}_{pool_names[idx]}", P_pool[idx]))

            # =========================
            # 物理验证与动态剔除
            # =========================
            print(f"[AL Manager] 调用 solve_ivp 物理验证 {len(new_params_list)} 个精选刺客细胞...")
            valid_results = []
            
            with ProcessPoolExecutor(max_workers=32) as executor:
                futures = [executor.submit(solve_single_mutant_bridge, name, p) for name, p in new_params_list]
                for f in futures:
                    res = f.result()
                    if res is not None: 
                        valid_results.append(res)
            
            if len(valid_results) > 0:
                Y_train_norm = dm.train_dataset_lhs.tensors[2].cpu()
                std_y_cpu = dm.std_y.cpu()
                mean_y_cpu = dm.mean_y.cpu()
                Y_train_raw = Y_train_norm * std_y_cpu + mean_y_cpu
                
                real_envelope = Y_train_raw.max(dim=-1)[0].max(dim=0)[0].numpy()
                dynamic_thresholds = np.maximum(real_envelope * 2.0, 2.0)
                
                pruned_results = []
                concentration_drop_count = 0
                
                for res in valid_results:
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
                        
                if concentration_drop_count > 0:
                    print(f"🛡️ [AL Manager] 动态浓度剔除系统销毁了 {concentration_drop_count} 个动态超限细胞！")
                
                valid_results = pruned_results

            # =========================
            # 安全合并与数据广播
            # =========================
            if len(valid_results) > 0:
                print(f"[AL Manager] 最终存活并允许入库的高难细胞: {len(valid_results)} 个")
                broadcast_data = {
                    'X': torch.stack([torch.tensor(r[0]) for r in valid_results]).float(),
                    'Y': torch.stack([torch.tensor(r[1]) for r in valid_results]).float(),
                    'P': torch.stack([torch.tensor(r[2]) for r in valid_results]).float(),
                    'names': [r[0] for r in new_params_list][:len(valid_results)],
                    'labels': [r[3] for r in valid_results]
                }

        if trainer.world_size > 1:
            broadcast_data = trainer.strategy.broadcast(broadcast_data, src=0)

        if broadcast_data is not None:
            from torch.utils.data import TensorDataset
            
            old_X, old_P, old_Y = dm.train_dataset_lhs.tensors
            
            updated_X = torch.cat([old_X, broadcast_data['X'].to(old_X.device)], dim=0)
            updated_P = torch.cat([old_P, broadcast_data['P'].to(old_P.device)], dim=0)
            updated_Y = torch.cat([old_Y, broadcast_data['Y'].to(old_Y.device)], dim=0)
            
            dm.train_dataset_lhs = TensorDataset(updated_X, updated_P, updated_Y)
            print(f"📦 [AL Manager] 训练集成功扩容！当前新样本总数: {len(dm.train_dataset_lhs)}")