import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR

import math

from .components.pure_fno import PureFNO_ICLR2021
from .components.cross_fno import NIPSYeastFNO
from .components.ultimate_fno import UltimateFNO
from .components.wno_cross import NIPSYeastWNO
import os
import numpy as np

from budding_yeast_v2.models.components import (
    PureFNO_ICLR2021, PureWNO_2022, NIPSYeastFNO, UltimateFNO,
    NIPSYeastWNO, NIPSYeastLNO, HyperFNO, ChannelHyperFNO,
    ResNetFiLM, DeepONet, hpYeastFNO, NeuralODEModel, GraphResNetFiLM,
    GatedGraphResNetFiLM, MultiScaleDiffGraphNet, DualPathSpectralGraphNet,
    BioStateSpaceGraphNet, GCGNetYeast,
    BioHeteroCausalGraphNet, HeteroGraphResNetFiLM,
    HeteroGraphDualPathNet, HeteroGraphChannelHyperFNO,
    DynamicReactionFluxOperator, SpectralTemporalHybridOperator,
    PhysicsConstrainedBiochemicalOperator,
    BioMechanisticOperator, BioMechFull,
    BioMechEventOnly, BioMechTitrationOnly, BioMechPathwayOnly,
    BioRefiner, BioRefinerCurvature,
    BioRefinerPyramid, BioRefinerGraph, BioRefinerPlus,
    BioRefinerBifurcate, BioRefinerGraphFlux, BioRefinerMechanistic,
    BioRefinerCausalBase, BioRefinerCausalCurvature,
    BioRefinerCausalPyramid, BioRefinerCausalGraph,
    BioRefinerCausalPlus, BioRefinerCausal,
    BioRefinerDynamicCausalPlus,
    BioRefinerLatchCausalPlus,
    HANO, HANO_NoEvent, HANO_NoFlux, HANO_NoGraph, HANO_Refine, HANO_Dual,
    FNO1d_Standard, UNet1d, GalerkinTransformer1d,
    CNO1d, MWT1d, Transolver1d, DPOT1d
)
from .components.bio_refiner_dynamic_plus import BioRefinerDynamicPlus
from .components.bio_event_refiner import BioEventHybridRefiner

from budding_yeast_v2.utils.losses import LOSS_REGISTRY, get_curriculum_weights


MODEL_REGISTRY = {
    'pure_fno': PureFNO_ICLR2021,
    'pure_wno': PureWNO_2022,
    'fno1d': FNO1d_Standard,
    'fno1d_standard': FNO1d_Standard,
    'unet1d': UNet1d,
    'unet': UNet1d,
    'galerkin_transformer1d': GalerkinTransformer1d,
    'galerkin_transformer': GalerkinTransformer1d,
    'galerkin': GalerkinTransformer1d,
    'cno1d': CNO1d,
    'cno': CNO1d,
    'mwt1d': MWT1d,
    'mwt': MWT1d,
    'transolver1d': Transolver1d,
    'transolver': Transolver1d,
    'dpot1d': DPOT1d,
    'dpot': DPOT1d,
    'cross_fno': NIPSYeastFNO,
    'ultimate_fno': UltimateFNO,
    'wno': NIPSYeastWNO,
    'lno': NIPSYeastLNO,
    'hyper_fno': HyperFNO,
    'channel_hyper_fno': ChannelHyperFNO,
    'resnet_film': ResNetFiLM,
    'deeponet': DeepONet,
    'hp_fno': hpYeastFNO,
    'neural_ode': NeuralODEModel,
    'graph_resnet_film': GraphResNetFiLM,
    'gated_graph_resnet_film': GatedGraphResNetFiLM,
    'multiscale_diffgraph': MultiScaleDiffGraphNet,
    'dualpath_spectralgraph': DualPathSpectralGraphNet,
    'bio_statespace_graph': BioStateSpaceGraphNet,
    'gcgnet': GCGNetYeast,
    'bio_hetero_causal_graph': BioHeteroCausalGraphNet,
    'hetero_graph_resnet_film': HeteroGraphResNetFiLM,
    'hetero_graph_dualpath': HeteroGraphDualPathNet,
    'hetero_graph_channel_hyper_fno': HeteroGraphChannelHyperFNO,
    'dyrfo': DynamicReactionFluxOperator,
    'dynamic_reaction_flux_operator': DynamicReactionFluxOperator,
    'st_hyperno': SpectralTemporalHybridOperator,
    'spectral_temporal_hybrid_operator': SpectralTemporalHybridOperator,
    'pc_bio_operator': PhysicsConstrainedBiochemicalOperator,
    'physics_constrained_biochemical_operator': PhysicsConstrainedBiochemicalOperator,
    'biom_full': BioMechFull,
    'biom_net': BioMechFull,
    'bio_mechanistic_operator': BioMechFull,
    'biom_event': BioMechEventOnly,
    'biom_titration': BioMechTitrationOnly,
    'biom_pathway': BioMechPathwayOnly,
    'biom_refiner': BioRefiner,
    'bio_refiner': BioRefiner,
    'biom_refiner_curvature': BioRefinerCurvature,
    'biom_refiner_curv': BioRefinerCurvature,
    'biom_refiner_pyramid': BioRefinerPyramid,
    'biom_refiner_pyr': BioRefinerPyramid,
    'biom_refiner_graph': BioRefinerGraph,
    'biom_refiner_graph_attn': BioRefinerGraph,
    'biom_refiner_plus': BioRefinerPlus,
    'biom_refiner_bifurcate': BioRefinerBifurcate,
    'biom_refiner_graph_flux': BioRefinerGraphFlux,
    'biom_refiner_mechanistic': BioRefinerMechanistic,
    'biom_refiner_causal_base': BioRefinerCausalBase,
    'biom_refiner_causal_curvature': BioRefinerCausalCurvature,
    'biom_refiner_causal_curv': BioRefinerCausalCurvature,
    'biom_refiner_causal_pyramid': BioRefinerCausalPyramid,
    'biom_refiner_causal_pyr': BioRefinerCausalPyramid,
    'biom_refiner_causal_graph': BioRefinerCausalGraph,
    'biom_refiner_causal_graph_attn': BioRefinerCausalGraph,
    'biom_refiner_causal_plus': BioRefinerCausalPlus,
    'biom_refiner_causal': BioRefinerCausal,
    'biom_refiner_dynamic_causal_plus': BioRefinerDynamicCausalPlus,
    'biom_refiner_dynamic_causal': BioRefinerDynamicCausalPlus,
    'biom_refiner_latch_causal_plus': BioRefinerLatchCausalPlus,
    'biom_refiner_latch_causal': BioRefinerLatchCausalPlus,
    'biom_refiner_dynamic_plus': BioRefinerDynamicPlus,
    'biom_refiner_event_hybrid': BioEventHybridRefiner,
    'bio_event_hybrid_refiner': BioEventHybridRefiner,
    'hano': HANO,
    'hano_no_event': HANO_NoEvent,
    'hano_no_flux': HANO_NoFlux,
    'hano_no_graph': HANO_NoGraph,
    'hano_refine': HANO_Refine,
    'hano_dual': HANO_Dual,
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
        # PINO 损失相关超参数 (loss_type='pino' 时生效)
        lambda_h1=0.1,
        h1_beta=1.0,
        # 机制增强与分岔损失超参数 (方案 A, 方案 C, 方案 A+C)
        lambda_pattern=0.1,
        lambda_arrest=0.05,
        pino_event_strategy='decoupled',
        huber_delta=1.0,
        # 自回归滚动选项
        rollout_type='one_time',  # 'one_time' (默认, 一次性预测) 或 'autoregressive' (自回归滚动)
    ):
        super().__init__()
        self.save_hyperparameters()

        # 自动加载生物符号反应拓扑图 (如果可用且未手动传入)
        if adj_matrix is None and model_name in ('graph_resnet_film', 'gated_graph_resnet_film', 'multiscale_diffgraph', 'dualpath_spectralgraph', 'bio_statespace_graph', 'cross_fno', 'channel_hyper_fno', 'biom_refiner_graph', 'biom_refiner_graph_attn', 'biom_refiner_plus', 'biom_refiner_bifurcate', 'biom_refiner_graph_flux', 'biom_refiner_mechanistic', 'biom_refiner_causal_graph', 'biom_refiner_causal_graph_attn', 'biom_refiner_causal_plus', 'biom_refiner_causal', 'biom_refiner_dynamic_causal_plus', 'biom_refiner_dynamic_causal', 'biom_refiner_dynamic_plus', 'biom_refiner_event_hybrid', 'bio_event_hybrid_refiner'):
            adj_path = 'budding_yeast_v2/data/yeast_signed_adjacency_matrix.npy'
            if os.path.exists(adj_path):
                adj_matrix = np.load(adj_path)

        ModelClass = MODEL_REGISTRY[model_name]

        if model_name == 'ultimate_fno':
            self.model = ModelClass(causal_matrix=causal_matrix, param_dim=param_dim, modes=modes)
        elif model_name == 'cross_fno':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, modes=modes,
                                    width=width, n_blocks=n_blocks, adj_matrix=adj_matrix)
        elif model_name == 'pure_fno':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, modes=modes, width=width)
        elif model_name in ('fno1d', 'fno1d_standard'):
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, modes=modes,
                                    width=width, n_blocks=n_blocks)
        elif model_name in ('unet1d', 'unet'):
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, width=width,
                                    n_blocks=n_blocks)
        elif model_name in ('galerkin_transformer1d', 'galerkin_transformer', 'galerkin'):
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, width=width,
                                    n_blocks=n_blocks)
        elif model_name in ('cno1d', 'cno'):
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, width=width,
                                    n_blocks=n_blocks)
        elif model_name in ('mwt1d', 'mwt'):
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, width=width,
                                    n_blocks=n_blocks)
        elif model_name in ('transolver1d', 'transolver'):
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, width=width,
                                    n_blocks=n_blocks)
        elif model_name in ('dpot1d', 'dpot'):
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, hidden_size=width,
                                    modes=modes, depth=n_blocks)
        elif model_name == 'pure_wno':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, width=width,
                                    n_blocks=n_blocks, wavelet=wavelet)
        elif model_name == 'wno':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, width=width,
                                    n_blocks=n_blocks, adj_matrix=adj_matrix, wavelet=wavelet)
        elif model_name == 'lno':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, modes=modes,
                                    width=width, n_blocks=n_blocks, adj_matrix=adj_matrix)
        elif model_name == 'hyper_fno':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, modes=modes,
                                    width=width, n_layers=n_blocks, d_lambda=d_lambda)
        elif model_name == 'channel_hyper_fno':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, modes=modes,
                                    width=width, n_layers=n_blocks, d_lambda=d_lambda,
                                    adj_matrix=adj_matrix)
        elif model_name == 'resnet_film':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim,
                                    width=width, n_blocks=n_blocks, d_lambda=d_lambda)
        elif model_name in ('graph_resnet_film', 'gated_graph_resnet_film', 'multiscale_diffgraph', 'bio_statespace_graph'):
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim,
                                    width=width, n_blocks=n_blocks, d_lambda=d_lambda,
                                    adj_matrix=adj_matrix)
        elif model_name == 'dualpath_spectralgraph':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim,
                                    width=width, modes=modes, n_blocks=n_blocks, d_lambda=d_lambda,
                                    adj_matrix=adj_matrix)
        elif model_name == 'deeponet':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim,
                                    p_basis=width, hidden_dim=width * 2)
        elif model_name == 'hp_fno':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, modes=modes, width=width)
        elif model_name == 'neural_ode':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, hidden_dim=width,
                                    method=ode_method, rtol=ode_rtol, atol=ode_atol)
        elif model_name == 'gcgnet':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, t_steps=500,
                                    patch_len=25, width=width, d_ff=width * 2,
                                    n_heads=4, e_layers=n_blocks if n_blocks > 0 else 2,
                                    rank=16)
        elif model_name in ('bio_hetero_causal_graph', 'hetero_graph_resnet_film'):
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim,
                                    width=width, n_blocks=n_blocks, d_lambda=d_lambda)
        elif model_name in ('hetero_graph_dualpath', 'hetero_graph_channel_hyper_fno'):
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim,
                                    modes=modes, width=width, n_blocks=n_blocks, d_lambda=d_lambda)
        elif model_name in ('dyrfo', 'dynamic_reaction_flux_operator', 'pc_bio_operator', 'physics_constrained_biochemical_operator'):
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim,
                                    width=width, n_blocks=n_blocks)
        elif model_name in ('st_hyperno', 'spectral_temporal_hybrid_operator',
                            'biom_full', 'biom_net', 'bio_mechanistic_operator',
                            'biom_event', 'biom_titration', 'biom_pathway',
                            'biom_refiner', 'bio_refiner',
                            'biom_refiner_curvature', 'biom_refiner_curv',
                            'biom_refiner_pyramid', 'biom_refiner_pyr',
                            'biom_refiner_causal_base',
                            'biom_refiner_causal_curvature', 'biom_refiner_causal_curv',
                            'biom_refiner_causal_pyramid', 'biom_refiner_causal_pyr'):
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim,
                                    modes=modes, width=width, n_blocks=n_blocks)
        elif model_name in ('biom_refiner_graph', 'biom_refiner_graph_attn', 'biom_refiner_plus',
                            'biom_refiner_bifurcate', 'biom_refiner_graph_flux', 'biom_refiner_mechanistic',
                            'biom_refiner_causal_graph', 'biom_refiner_causal_graph_attn',
                            'biom_refiner_causal_plus', 'biom_refiner_causal',
                            'biom_refiner_dynamic_causal_plus', 'biom_refiner_dynamic_causal',
                            'biom_refiner_latch_causal_plus', 'biom_refiner_latch_causal',
                            'biom_refiner_dynamic_plus',
                            'biom_refiner_event_hybrid', 'bio_event_hybrid_refiner'):
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim,
                                    modes=modes, width=width, n_blocks=n_blocks,
                                    adj_matrix=adj_matrix)
        elif model_name in ('hano', 'hano_no_event', 'hano_no_flux', 'hano_no_graph', 'hano_refine', 'hano_dual'):
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim,
                                    width=width, n_blocks=n_blocks)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        loss_class = LOSS_REGISTRY[loss_type]
        if loss_type in ('mse_only', 'mse_smooth', 'h1', 'rel_l2', 'hybrid_mse_rel',
                         'phase_derivative', 'soft_dtw', 'hybrid_phase_deriv', 'hybrid_soft_dtw',
                         'triple_dynamic', 'hybrid_phase_dtw'):
            self.criterion = loss_class()
        elif loss_type == 'hybrid_bifurcate':
            self.criterion = loss_class(lambda_pattern=lambda_pattern, lambda_arrest=lambda_arrest)
        elif loss_type in ('pinn_residual', 'pino', 'hybrid_pino_colloc', 'hybrid_all'):
            # 创建占位实例以在 ModelSummary 中显示损失类型
            # 真实参数在 on_train_start 中用 datamodule 统计量重建
            dummy_mean = torch.zeros(1, num_vars, 1)
            dummy_std = torch.ones(1, num_vars, 1)
            dummy_p_mean = torch.zeros(1, param_dim)
            dummy_p_std = torch.ones(1, param_dim)
            pinn_kwargs = {
                't_max': 210.0,
                'lambda_phys': lambda_phys,
                'lambda_jump': lambda_jump,
                'lambda_penalty': lambda_penalty,
                'event_window': event_window,
                'event_tol': event_tol,
                'jump_ratio': jump_ratio,
                'residual_subsample': residual_subsample,
                'event_weight': event_weight,
            }
            if loss_type == 'pino':
                pinn_kwargs.update({'lambda_h1': lambda_h1, 'h1_beta': h1_beta})
            elif loss_type in ('hybrid_pino_colloc', 'hybrid_all'):
                pinn_kwargs.update({
                    'event_strategy': pino_event_strategy,
                    'huber_delta': huber_delta,
                })
                if loss_type == 'hybrid_all':
                    pinn_kwargs.update({
                        'lambda_pattern': lambda_pattern,
                        'lambda_arrest': lambda_arrest,
                    })
            self.criterion = loss_class(
                dummy_mean, dummy_std, dummy_p_mean, dummy_p_std,
                **pinn_kwargs
            )
        else:
            self.criterion = None

        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.adj_matrix = adj_matrix

    def setup(self, stage=None):
        # HANO 等结构保持模型需要在真实空间完成通量积分/事件跳变,
        # 注入 datamodule 归一化统计量 (persistent buffer, 随 checkpoint 持久化)
        self._inject_model_stats()

    def _inject_model_stats(self):
        if not hasattr(self.model, 'set_stats'):
            return
        dm = getattr(self.trainer, 'datamodule', None)
        if dm is None or not hasattr(dm, 'mean_y'):
            return
        try:
            self.model.set_stats(dm.mean_y, dm.std_y, dm.p_mean, dm.p_std, float(dm.t_max))
        except Exception as e:
            print(f"[YeastLitModule] 统计量注入失败 (将在 on_train_start 重试): {e}")

    def on_train_start(self):
        # 统计量注入兜底 (setup 时 datamodule 统计量可能尚未就绪)
        self._inject_model_stats()
        if self.trainer.datamodule is not None:
            dm = self.trainer.datamodule
            loss_type = self.hparams.loss_type
            loss_class = LOSS_REGISTRY[loss_type]
            if loss_type in ('pinn_residual', 'pino', 'hybrid_pino_colloc', 'hybrid_all'):
                # 用 datamodule 真实统计量重建（替换 __init__ 中的占位实例）
                pinn_kwargs = {
                    't_max': float(dm.t_max),
                    'lambda_phys': self.hparams.lambda_phys,
                    'lambda_jump': self.hparams.lambda_jump,
                    'lambda_penalty': self.hparams.lambda_penalty,
                    'event_window': self.hparams.event_window,
                    'event_tol': self.hparams.event_tol,
                    'jump_ratio': self.hparams.jump_ratio,
                    'residual_subsample': self.hparams.residual_subsample,
                    'event_weight': self.hparams.event_weight,
                }
                if loss_type == 'pino':
                    pinn_kwargs.update({'lambda_h1': self.hparams.lambda_h1, 'h1_beta': self.hparams.h1_beta})
                elif loss_type in ('hybrid_pino_colloc', 'hybrid_all'):
                    pinn_kwargs.update({
                        'event_strategy': self.hparams.pino_event_strategy,
                        'huber_delta': self.hparams.huber_delta,
                    })
                    if loss_type == 'hybrid_all':
                        pinn_kwargs.update({
                            'lambda_pattern': self.hparams.lambda_pattern,
                            'lambda_arrest': self.hparams.lambda_arrest,
                        })
                self.criterion = loss_class(
                    dm.mean_y.to(self.device),
                    dm.std_y.to(self.device),
                    dm.p_mean.to(self.device),
                    dm.p_std.to(self.device),
                    **pinn_kwargs
                )
            elif loss_type == 'hybrid_bifurcate':
                self.criterion = loss_class(
                    lambda_pattern=self.hparams.lambda_pattern,
                    lambda_arrest=self.hparams.lambda_arrest
                )
            elif self.criterion is None:
                if loss_type in ('mse_only', 'mse_smooth', 'h1', 'rel_l2', 'hybrid_mse_rel',
                                 'phase_derivative', 'soft_dtw', 'hybrid_phase_deriv', 'hybrid_soft_dtw',
                                 'triple_dynamic', 'hybrid_phase_dtw'):
                    self.criterion = loss_class()
                else:
                    self.criterion = loss_class(
                        dm.mean_y.to(self.device),
                        dm.std_y.to(self.device),
                    )

    def forward(self, x, p):
        return self.forward_ic_time(x, p)

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
        if len(batch) == 4:
            x, p, y, pat_id = batch
        else:
            x, p, y = batch
            pat_id = None

        curr_ep = getattr(self, 'current_epoch', 0)
        lam_pen, lam_sm = get_curriculum_weights(curr_ep, self.max_epochs)

        weights = torch.ones(len(x), dtype=torch.float32, device=self.device)

        if self.hparams.model_name.startswith('hano') and self.training:
            out_full, aux = self.model(x, p, return_aux=True)
            if self.hparams.loss_type in ('pinn_residual', 'pino', 'hybrid_pino_colloc', 'hybrid_all'):
                self.criterion.set_epoch(curr_ep)
                loss = self.criterion(out_full, y, p, weights, lam_pen, lam_sm)
            else:
                loss = self.criterion(out_full, y, weights, lam_pen, lam_sm)
                # 双路径融合: 物理路径独立辅助损失 (防止门控偏向直接路径后物理分支梯度饿死)
                if 'physics_out' in aux:
                    phys_loss = self.criterion(aux['physics_out'], y, weights, lam_pen, lam_sm)
                    loss = loss + self.model.lambda_phys * phys_loss
                    self.log('train_phys_loss', phys_loss, on_step=False, on_epoch=True,
                             prog_bar=True, sync_dist=True)
            # 事件辅助损失: 事件时钟热图 logits vs 从目标轨迹自动提取的事件标签 (免费监督)
            if 'event_logits' in aux:
                labels = self.model.compute_event_labels(y)          # (B, 4, T)
                ev_loss = F.binary_cross_entropy_with_logits(aux['event_logits'], labels)
                loss = loss + self.model.lambda_event * ev_loss
                self.log('train_event_loss', ev_loss, on_step=False, on_epoch=True,
                         prog_bar=True, sync_dist=True)
        elif self.hparams.model_name.startswith(('biom_refiner', 'bio_refiner')) and self.training:
            pattern_logits = None
            if hasattr(self.model, 'pattern_head'):
                out_full, out_coarse, pattern_logits = self.model(x, p, return_coarse=True, return_pattern=True)
            else:
                out_full, out_coarse = self.model(x, p, return_coarse=True)

            if self.hparams.loss_type == 'hybrid_bifurcate':
                loss_final = self.criterion(out_full, y, pattern_logits=pattern_logits, pattern_targets=pat_id)
                loss_coarse = self.criterion(out_coarse, y, pattern_logits=None, pattern_targets=None)
                loss = loss_final + 0.3 * loss_coarse
            elif self.hparams.loss_type == 'hybrid_pino_colloc':
                self.criterion.set_epoch(curr_ep)
                loss_final = self.criterion(out_full, y, p, weights, lam_pen, lam_sm)
                loss_coarse = self.criterion(out_coarse, y, p, weights, lam_pen, lam_sm)
                loss = loss_final + 0.3 * loss_coarse
            elif self.hparams.loss_type == 'hybrid_all':
                self.criterion.set_epoch(curr_ep)
                loss_final = self.criterion(out_full, y, p, weights, lam_pen, lam_sm,
                                            pattern_logits=pattern_logits, pattern_targets=pat_id)
                loss_coarse = self.criterion(out_coarse, y, p, weights, lam_pen, lam_sm,
                                             pattern_logits=None, pattern_targets=None)
                loss = loss_final + 0.3 * loss_coarse
            elif self.hparams.loss_type in ('pinn_residual', 'pino'):
                self.criterion.set_epoch(curr_ep)
                loss_final = self.criterion(out_full, y, p, weights, lam_pen, lam_sm)
                loss_coarse = self.criterion(out_coarse, y, p, weights, lam_pen, lam_sm)
                loss = loss_final + 0.3 * loss_coarse
            else:
                loss_final = self.criterion(out_full, y, weights, lam_pen, lam_sm)
                loss_coarse = self.criterion(out_coarse, y, weights, lam_pen, lam_sm)
                loss = loss_final + 0.3 * loss_coarse
        else:
            out_full = self.forward_ic_time(x, p)

            if self.hparams.loss_type in ('pinn_residual', 'pino', 'hybrid_pino_colloc', 'hybrid_all'):
                self.criterion.set_epoch(curr_ep)
                loss = self.criterion(out_full, y, p, weights, lam_pen, lam_sm)
                if hasattr(self.criterion, '_last_loss_components'):
                    for k, v in self.criterion._last_loss_components.items():
                        self.log(f'train_{k}', v, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            elif self.hparams.loss_type == 'hybrid_bifurcate':
                loss = self.criterion(out_full, y, pattern_targets=pat_id)
            else:
                loss = self.criterion(out_full, y, weights, lam_pen, lam_sm)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x_v, p_v, y_v = batch[:3]
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