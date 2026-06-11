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
from budding_yeast_v2.utils.metrics import calculate_metrics


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
        else:
            self.criterion = None

        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.adj_matrix = adj_matrix

    def on_train_start(self):
        if self.trainer.datamodule is not None:
            dm = self.trainer.datamodule
            if self.criterion is None:
                loss_type = self.hparams.loss_type
                loss_class = LOSS_REGISTRY[loss_type]
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