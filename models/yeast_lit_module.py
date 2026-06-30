import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR

import math

from .components.pure_fno import PureFNO_ICLR2021
from .components.cross_fno import NIPSYeastFNO
from .components.ultimate_fno import UltimateFNO
from .components.wno_cross import NIPSYeastWNO
from .components.lno_cross import NIPSYeastLNO
from .components.hyper_fno import HyperFNO
from .components.hp_fno import hpYeastFNO
from .components.neural_ode import NeuralODEModel
from .components.patchtst import PatchTST
from .components.unet1d_adapter import Unet1DAdapter
from .components.fno1d_adapter import FNO1DAdapter

from budding_yeast_v2.utils.losses import LOSS_REGISTRY, get_curriculum_weights


MODEL_REGISTRY = {
    'pure_fno': PureFNO_ICLR2021,
    'cross_fno': NIPSYeastFNO,
    'ultimate_fno': UltimateFNO,
    'wno': NIPSYeastWNO,
    'lno': NIPSYeastLNO,
    'hyper_fno': HyperFNO,
    'hp_fno': hpYeastFNO,
    'neural_ode': NeuralODEModel,
    'patchtst': PatchTST,
    'unet1d': Unet1DAdapter,
    'fno1d': FNO1DAdapter,
}


class YeastLitModule(pl.LightningModule):
    def __init__(
        self,
        model_name='pure_fno',
        num_vars=39,
        param_dim=141,
        modes=24,
        width=64,
        n_blocks=4,
        adj_matrix=None,
        causal_matrix=None,
        loss_type='physics_informed',
        lr=2e-4,
        weight_decay=5e-2,
        max_epochs=120,
        warmup_epochs=10,
        wavelet='haar',
        d_lambda=32,
        ode_method='dopri5',
        ode_rtol=1e-3,
        ode_atol=1e-4,
        # PINN 残差损失相关超参数
        lambda_phys=0.1,
        lambda_jump=1.0,
        lambda_penalty=1.0,
        event_window=4,
        event_tol=0.08,
        jump_ratio=0.3,
        residual_subsample=1,
        event_weight=0.1,
        # 自回归滚动选项
        rollout_type='one_time',  # 'one_time' (默认, 一次性预测) 或 'autoregressive' (自回归滚动)
    ):
        super().__init__()
        self.save_hyperparameters()

        ModelClass = MODEL_REGISTRY[model_name]

        if model_name == 'ultimate_fno':
            self.model = ModelClass(causal_matrix=causal_matrix, param_dim=param_dim, modes=modes)
        elif model_name == 'cross_fno':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, modes=modes,
                                    width=width, n_blocks=n_blocks, adj_matrix=adj_matrix)
        elif model_name == 'pure_fno':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, modes=modes, width=width)
        elif model_name == 'wno':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, width=width,
                                    n_blocks=n_blocks, adj_matrix=adj_matrix, wavelet=wavelet)
        elif model_name == 'lno':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, modes=modes,
                                    width=width, n_blocks=n_blocks, adj_matrix=adj_matrix)
        elif model_name == 'hyper_fno':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, modes=modes,
                                    width=width, n_layers=n_blocks, d_lambda=d_lambda)
        elif model_name == 'hp_fno':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, modes=modes, width=width)
        elif model_name == 'neural_ode':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, hidden_dim=width,
                                    method=ode_method, rtol=ode_rtol, atol=ode_atol)
        elif model_name == 'patchtst':
            # PatchTST (ICLR 2023): 通道独立 + Patching + Transformer, 适合自回归滚动
            # modes 参数在此用作兼容接口（未使用），width 用作 d_model
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, modes=modes,
                                    width=width, n_blocks=n_blocks)
        elif model_name == 'unet1d':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim,
                                    hidden_channels=width, n_blocks=n_blocks)
        elif model_name == 'fno1d':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim,
                                    modes=modes, width=width)

        loss_class = LOSS_REGISTRY[loss_type]
        if loss_type == 'mse_only':
            self.criterion = loss_class()
        elif loss_type == 'mse_smooth':
            self.criterion = loss_class()
        elif loss_type == 'pinn_residual':
            # 创建占位实例以在 ModelSummary 中显示损失类型
            # 真实参数在 on_train_start 中用 datamodule 统计量重建
            dummy_mean = torch.zeros(1, num_vars, 1)
            dummy_std = torch.ones(1, num_vars, 1)
            dummy_p_mean = torch.zeros(1, param_dim)
            dummy_p_std = torch.ones(1, param_dim)
            self.criterion = loss_class(
                dummy_mean, dummy_std, dummy_p_mean, dummy_p_std,
                t_max=210.0, lambda_phys=lambda_phys, lambda_jump=lambda_jump,
                event_window=event_window, event_tol=event_tol,
                jump_ratio=jump_ratio, residual_subsample=residual_subsample,
                event_weight=event_weight,
            )
        else:
            self.criterion = None

        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.adj_matrix = adj_matrix

    def on_train_start(self):
        if self.trainer.datamodule is not None:
            dm = self.trainer.datamodule
            loss_type = self.hparams.loss_type
            loss_class = LOSS_REGISTRY[loss_type]
            if loss_type == 'pinn_residual':
                # 用 datamodule 真实统计量重建（替换 __init__ 中的占位实例）
                self.criterion = loss_class(
                    dm.mean_y.to(self.device),
                    dm.std_y.to(self.device),
                    dm.p_mean.to(self.device),
                    dm.p_std.to(self.device),
                    t_max=float(dm.t_max),
                    lambda_phys=self.hparams.lambda_phys,
                    lambda_jump=self.hparams.lambda_jump,
                    lambda_penalty=self.hparams.lambda_penalty,
                    event_window=self.hparams.event_window,
                    event_tol=self.hparams.event_tol,
                    jump_ratio=self.hparams.jump_ratio,
                    residual_subsample=self.hparams.residual_subsample,
                    event_weight=self.hparams.event_weight,
                )
            elif self.criterion is None:
                if loss_type in ('mse_only', 'mse_smooth'):
                    self.criterion = loss_class()
                else:
                    self.criterion = loss_class(
                        dm.mean_y.to(self.device),
                        dm.std_y.to(self.device),
                    )

    def forward_ic_time(self, x, p):
        """
        前向传播，根据 rollout_type 选择一次性预测或自回归滚动。

        Args:
            x: (B, V, 2, T) 输入张量，dim 2 = [IC, t_grid]
            p: (B, param_dim) 参数向量

        Returns:
            (B, V, T) 预测轨迹
        """
        if self.hparams.model_name == 'ultimate_fno':
            # ultimate_fno 有特殊的双输入接口，不支持自回归滚动
            return self.model(x[:, :, 0, :], x[:, 0:1, 1, :], p)

        if self.hparams.rollout_type == 'autoregressive':
            return self._autoregressive_rollout(x, p)
        else:
            # 默认: 一次性预测（保留原有逻辑）
            return self.model(x, p)

    def _autoregressive_rollout(self, x, p):
        """
        自回归滚动预测：将当前步输出反馈为下一步输入，逐步生成完整轨迹。

        核心思想：算子 G 不断把当前步输出当成下一步输入，一步步迭代。
        为兼容频域算子对序列长度的要求，每次前向传播都使用全尺寸 (B, V, 2, T) 输入，
        仅提取输出中索引 1（下一步）的结果。

        Args:
            x: (B, V, 2, T) 输入张量，x[:, :, 0, :] = IC, x[:, :, 1, :] = t_grid
            p: (B, param_dim) 参数向量

        Returns:
            trajectory: (B, V, T) 自回归生成的完整轨迹
        """
        B, V, _, T = x.shape

        # 初始化轨迹张量，存放最终结果
        trajectory = torch.zeros(B, V, T, device=x.device, dtype=x.dtype)

        # 第 0 步：使用输入中的初始条件
        trajectory[:, :, 0] = x[:, :, 0, 0]  # (B, V)

        # 保留原始 t_grid 通道（不修改）
        t_grid = x[:, :, 1, :]  # (B, V, T)

        # 逐步滚动：从第 0 步预测第 1 步，第 1 步预测第 2 步，...
        for k in range(T - 1):
            # 克隆全尺寸临时张量，保持频域算子所需的序列长度
            temp_input = x.clone()
            # 将当前步状态广播铺满 IC 通道
            current_state = trajectory[:, :, k].unsqueeze(-1)  # (B, V, 1)
            temp_input[:, :, 0, :] = current_state.expand(-1, -1, T)  # (B, V, T)
            # t_grid 通道保持不变（已在 clone 中保留）

            # 前向传播，得到全尺寸预测
            out = self.model(temp_input, p)  # (B, V, T)

            # 仅提取下一步（索引 1）的结果，存入轨迹的 k+1 位置
            trajectory[:, :, k + 1] = out[:, :, 1]

        return trajectory

    def training_step(self, batch, batch_idx):
        x, p, y = batch

        lam_pen, lam_sm = get_curriculum_weights(self.current_epoch, self.max_epochs)

        weights = torch.ones(len(x), dtype=torch.float32, device=self.device)

        out_full = self.forward_ic_time(x, p)

        if self.hparams.loss_type == 'pinn_residual':
            # Curriculum learning 由 loss 类内部管理（set_epoch + _get_physics_weight）
            # v2: warmup_epochs=30（前30轮纯MSE），之后30轮线性增长到目标权重
            self.criterion.set_epoch(self.current_epoch)

            loss = self.criterion(out_full, y, p, weights, lam_pen, lam_sm)

            # 记录分量损失
            if hasattr(self.criterion, '_last_loss_components'):
                for k, v in self.criterion._last_loss_components.items():
                    self.log(f'train_{k}', v, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        else:
            loss = self.criterion(out_full, y, weights, lam_pen, lam_sm)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x_v, p_v, y_v = batch
        pred = self.forward_ic_time(x_v, p_v)
        target = y_v
        mse_val = F.mse_loss(pred, target)

        if dataloader_idx == 0:
            self.log('val_loss_lhs', mse_val, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False, sync_dist=True)
        elif dataloader_idx == 1:
            self.log('test_loss_lhs', mse_val, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False, sync_dist=True)
        elif dataloader_idx == 2:
            self.log('test_loss_real', mse_val, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False, sync_dist=True)

        return mse_val

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        warmup_epochs = self.hparams.get('warmup_epochs', 10)
        max_epochs = self.max_epochs
        eta_min = 1e-6

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # 线性 warmup: 0 → 1
                return epoch / max(warmup_epochs, 1)
            # Cosine annealing: 1 → eta_min/lr
            progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
            return eta_min / self.lr + 0.5 * (1.0 - eta_min / self.lr) * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            },
        }