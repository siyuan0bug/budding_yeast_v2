"""
Main entry point for AR + AL training.
Follows AL4PDE's run_active_learning.py logic, adapted for yeast dataset.
"""

import os
import time
import argparse
import yaml
import torch
import numpy as np

import wandb
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm

from budding_yeast_v2.data.param_gen import PDEParamGenerator, YeastSimulator, save_ar_batch
from budding_yeast_v2.data.ar_datamodule import ARDataModule
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.models.ar_model import ARModelWrapper

# Import our ported AL4PDE acquisition strategies
# Currently using simple Random selection as baseline
from budding_yeast_v2.acquisition.random_selection import RandomSelection
from budding_yeast_v2.acquisition.data_schedule import ExponentialSchedule, LinearSchedule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description="AR + AL Training for Budding Yeast")
    parser.add_argument('--config', type=str, default='configs/ar_al_default.yaml')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--num_al_iter', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--num_train_steps', type=int, default=None)
    parser.add_argument('--noise', type=float, default=None)
    parser.add_argument('--noise_mode', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--acquisition', type=str, default=None,
                        choices=['random', 'lcmd', 'max_dist', 'bait', 'kmeans++'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_wandb', action='store_true', default=False)
    return parser.parse_args()


def load_config(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    # Override with command line arguments
    if args.model is not None: cfg['model']['name'] = args.model
    if args.num_al_iter is not None: cfg['active_learning']['num_al_iter'] = args.num_al_iter
    if args.num_epochs is not None: cfg['training']['num_epochs'] = args.num_epochs
    if args.num_train_steps is not None: cfg['training']['num_train_steps'] = args.num_train_steps
    if args.noise is not None: cfg['training']['noise'] = args.noise
    if args.noise_mode is not None: cfg['training']['noise_mode'] = args.noise_mode
    if args.batch_size is not None: cfg['training']['batch_size'] = args.batch_size
    if args.acquisition is not None: cfg['active_learning']['acquisition'] = args.acquisition
    
    cfg['seed'] = args.seed
    cfg['no_wandb'] = args.no_wandb
    return cfg


def build_acquisition(cfg_al):
    schedule_type = cfg_al.get('data_schedule', 'exponential')
    first_batch = cfg_al.get('first_batch_num', 8)
    if schedule_type == 'exponential':
        schedule = ExponentialSchedule(first_batch)
    else:
        schedule = LinearSchedule(first_batch)
        
    acq_type = cfg_al.get('acquisition', 'random')
    if acq_type == 'random':
        return RandomSelection(
            schedule=schedule, 
            sim_batch_size=cfg_al.get('sim_batch_size', 16)
        )
    else:
        # Fallback to random for now (can map to others later)
        print(f"Warning: Acquisition {acq_type} not fully integrated. Using random.")
        return RandomSelection(schedule=schedule, sim_batch_size=cfg_al.get('sim_batch_size', 16))


def main():
    args = parse_args()
    cfg = load_config(args)
    
    # Setup directories
    save_dir = cfg.get('save_dir', './train_result_ar')
    run_name = cfg['wandb'].get('name')
    acq_type = cfg['active_learning'].get('acquisition', 'random')
    if run_name is None:
        run_name = f"{cfg['model']['name']}_ar_{cfg['training']['num_train_steps']}steps_al_{acq_type}_seed{cfg['seed']}"
    
    out_dir = os.path.join(save_dir, run_name)
    data_save_dir = os.path.join(out_dir, "data")
    ckpt_save_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(data_save_dir, exist_ok=True)
    os.makedirs(ckpt_save_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(out_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)
        
    # Setup Wandb
    if cfg['no_wandb']:
        os.environ["WANDB_MODE"] = "offline"
        print("WANDB offline mode enabled.")
    
    wandb.init(
        project=cfg['wandb'].get('project', 'budding_yeast_v2_ar_al'),
        name=run_name,
        group=cfg['wandb'].get('group', 'ar_al'),
        config=cfg
    )
    
    # Set seeds
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    
    # Initialize components
    param_gen = PDEParamGenerator(mode='lhs', seed=cfg['seed'])
    simulator = YeastSimulator(
        t_max=cfg['data'].get('t_max', 210), 
        t_steps=cfg['data'].get('t_steps', 500)
    )
    acq_type = cfg['active_learning'].get('acquisition', 'random')
    # acq_module will be built after the pool is generated
    data_module = ARDataModule(
        num_train_steps=cfg['training']['num_train_steps'],
        batch_size=cfg['training']['batch_size'],
        reduced_resolution_t=cfg['data']['reduced_resolution_t']
    )
    
    # Load real mutants directly from NPZ to get RAW data for evaluation
    print("Loading real mutants dataset for evaluation (RAW data)...")
    npz_data = np.load("/home/users/zsy/mydata/budding_yeast/lhs_blind_210min_500steps_dual_labels.npz", allow_pickle=True)
    mutant_names = npz_data['mutant_names']
    is_real = np.array([not str(name).startswith('LHS') for name in mutant_names])
    real_Y = torch.tensor(npz_data['data'][is_real], dtype=torch.float32)  # (N, C, T)
    real_P = torch.tensor(npz_data['params'][is_real], dtype=torch.float32)  # (N, P)
    
    def extract_pattern_label(name):
        if 'WT' in name: return 'Pattern_WT'
        if 'cln1' in name or 'cln2' in name or 'cln3' in name: return 'Pattern_G1'
        if 'clb' in name: return 'Pattern_S_G2_M'
        if 'cdc' in name: return 'Pattern_Arrest'
        return 'Pattern_Other'
        
    real_names = [str(n) for n in mutant_names[is_real]]
    real_labels = [extract_pattern_label(n) for n in real_names]
    
    class RawRealMutantLoader:
        def __init__(self, Y, P, names, labels, batch_size):
            self.Y = Y
            self.P = P
            self.names = names
            self.labels = labels
            self.batch_size = batch_size
        def __iter__(self):
            for i in range(0, len(self.Y), self.batch_size):
                y_batch = self.Y[i:i+self.batch_size]
                p_batch = self.P[i:i+self.batch_size]
                names_batch = self.names[i:i+self.batch_size]
                labels_batch = self.labels[i:i+self.batch_size]
                ic = y_batch[:, :, 0]
                t_idx = torch.zeros(y_batch.shape[0])
                yield ic, y_batch, p_batch, t_idx, names_batch, labels_batch
        def __len__(self):
            return (len(self.Y) + self.batch_size - 1) // self.batch_size

    # Use a large batch size for much faster evaluation (e.g. 512)
    real_mutant_loader = RawRealMutantLoader(real_Y, real_P, real_names, real_labels, batch_size=512)
    
    model = ARModelWrapper(cfg['model'], cfg['training'], t_steps=cfg['data']['t_steps'])
    
    # Track data folders
    data_folders = []
    
    # -------------------------------------------------------------------------
    # Initial Data Generation (And Candidate Pool Init)
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print(f"🚀 Phase 0: Generating Massive Candidate Pool (10000 per mutant)")
    print("="*50)
    pool_params, pool_dicts = param_gen.generate_massive_candidate_pool(samples_per_mutant=10000)
    print(f"✅ Generated {len(pool_dicts)} unique parameters for the candidate pool.")
    
    # init 阶段始终用 RandomPoolBased (初始无训练集, 无法做特征选择; 与 al4pde generate_initial 一致)
    from budding_yeast_v2.acquisition.pool_based_strategies import RandomPoolBased
    init_acq = RandomPoolBased(pool_params, pool_dicts)

    print("\n" + "="*50)
    print(f"🚀 Phase 1: Generating Initial Data from Pool (random)")
    print("="*50)

    init_batches = cfg['active_learning']['first_batch_num']
    sim_batch_size = cfg['active_learning']['sim_batch_size']
    init_folder = os.path.join(data_save_dir, "init")

    init_acq.generate(model, param_gen, simulator, "init", init_folder, batch_size=init_batches * sim_batch_size)

    data_folders.append(init_folder)

    # 构造 AL 循环用的 acquisition module
    # 复用 init 后剩余的 available_indices, 避免重复选池中样本
    if acq_type in ('lcmd', 'max_dist', 'bait', 'kmeans++'):
        from budding_yeast_v2.acquisition.lcmd_acquisition import LCMDAcquisition
        al_cfg = cfg['active_learning']
        acq_module = LCMDAcquisition(
            pool_params, pool_dicts,
            num_rollout_steps=al_cfg.get('num_rollout_steps', 20),
            pred_batch_size=al_cfg.get('pred_batch_size', 1024),
            sel_with_train=al_cfg.get('sel_with_train', True),
            sketch_dim=al_cfg.get('sketch_dim', 512),
            available_indices=init_acq.available_indices,
            selection_method=acq_type,
        )
        print(f"LCMDAcquisition initialized (method={acq_type}). Pool remaining: {len(acq_module.available_indices)}")
    else:
        acq_module = init_acq  # random 继续用同一个 RandomPoolBased
    
    # Build validation loader from a separate set (or use a slice of init data for simplicity)
    # For now, we evaluate on the training set to monitor fitting.
    # In a full run, we would load real_mutants here for the _val_loader.
    # To keep this script standalone, we'll evaluate on full trajectory rollouts of the train set.
    
    # -------------------------------------------------------------------------
    # Active Learning Loop
    # -------------------------------------------------------------------------
    total_epoch = 0
    num_al_iter = cfg['active_learning']['num_al_iter']
    num_epochs = cfg['training']['num_epochs']
    
    for al_iter in range(num_al_iter):
        print("\n" + "="*50)
        print(f"🔄 AL Iteration {al_iter+1} / {num_al_iter}")
        print("="*50)
        
        # 1. Load all data generated so far
        train_loader, train_loader_full, dataset = data_module.build_train_loader(data_folders)
        
        # 2. Init/Reset Model for this iteration (AL4PDE paradigm)
        # Pass real_mutant_loader as the validation loader to evaluate on real mutants
        model.init_training(al_iter, train_loader, train_loader_full, val_loader=real_mutant_loader)
        
        # 3. Train
        print(f"\n▶️ Training for {num_epochs} epochs...")
        model.train_n_epochs(al_iter, num_epochs, total_epoch, vis=True)
        total_epoch += num_epochs
        
        # 4. Save Checkpoint
        ckpt_path = os.path.join(ckpt_save_dir, f"model_al_{al_iter}.pt")
        save_dict = {
            'model_state': model.state_dict(),
            'task_norm_state': model.task_norm.state_dict(),
            'config': cfg
        }
        torch.save(save_dict, ckpt_path)
        print(f"✅ Checkpoint saved: {ckpt_path}")
        
        # 5. Generate New Data (if not last iter)
        if al_iter < num_al_iter - 1:
            print("\n▶️ Active Learning: Generating new batch from candidate pool...")
            iter_folder = os.path.join(data_save_dir, f"iter_{al_iter}")
            
            # The acquisition module selects from the pool, runs simulator, and saves
            sim_batch_size = cfg['active_learning']['sim_batch_size']
            if acq_type in ('lcmd', 'max_dist', 'bait', 'kmeans++'):
                acq_module.generate(model, param_gen, simulator, al_iter, iter_folder,
                                    batch_size=sim_batch_size, train_loader=train_loader)
            else:
                acq_module.generate(model, param_gen, simulator, al_iter, iter_folder, batch_size=sim_batch_size)
            data_folders.append(iter_folder)
            
    print("\n🎉 AR+AL Pipeline Finished Successfully!")
    
    # Run evaluation script
    print("\n" + "="*50)
    print("🚀 Auto-triggering Final Evaluation Script...")
    print("="*50)
    
    eval_result_dir = os.path.join(out_dir, "eval_result")
    # 释放 GPU 内存, 避免后续 eval 进程加载 checkpoint 时报 "CUDA-capable device(s) is/are busy"
    del model
    torch.cuda.empty_cache()
    # Using sys.executable to ensure the same Python environment
    import sys
    eval_cmd = (
        f"{sys.executable} scripts/eval_ar_al.py "
        f"--ckpt {ckpt_path} "
        f"--data_path /home/users/zsy/mydata/budding_yeast/lhs_blind_210min_500steps_dual_labels.npz "
        f"--save_dir {eval_result_dir} "
        f"--device {cfg['training'].get('device', 'cuda')}"
    )
    print(f"Executing: {eval_cmd}")
    os.system(eval_cmd)
    
    wandb.finish()

if __name__ == "__main__":
    main()
