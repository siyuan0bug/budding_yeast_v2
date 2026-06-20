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
        event_window=4,
        event_tol=0.08,
        jump_ratio=0.3,
        residual_subsample=1,
        event_weight=0.1,
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
        if self.hparams.model_name == 'ultimate_fno':
            return self.model(x[:, :, 0, :], x[:, 0:1, 1, :], p)
        else:
            return self.model(x, p)

    def training_step(self, batch, batch_idx):
        x, p, y = batch

        lam_pen, lam_sm = get_curriculum_weights(self.current_epoch, self.max_epochs)

        weights = torch.ones(len(x), dtype=torch.float32, device=self.device)

        out_full = self.forward_ic_time(x, p)

        if self.hparams.loss_type == 'pinn_residual':
            # PINN 残差损失需要归一化参数向量 p
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