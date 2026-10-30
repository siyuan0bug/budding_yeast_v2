import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR

import os
import math
import numpy as np
import copy
import warnings
from budding_yeast_v2.data.normalization import denormalize_y as decode_y
from budding_yeast_v2.data.preprocessing_contract import (
    validate_preprocessing_state, assert_compatible, resolve_checkpoint_norm,
    resolve_checkpoint_fit_scope,
)
from budding_yeast_v2.models.ours.graph_contracts import (
    resolve_variable_graph, set_graph_load_policy, variable_graph_report, fixed_graph_report,
)
from budding_yeast_v2.models.ours.relation_graph_contracts import configure_relation_view

from budding_yeast_v2.models.baselines import BASELINE_REGISTRY, build_baseline


from budding_yeast_v2.models.ours import (
    NIPSYeastFNO, UltimateFNO, NIPSYeastWNO, NIPSYeastLNO,
    HyperFNO, ChannelHyperFNO, hpYeastFNO, GraphResNetFiLM,
    GatedGraphResNetFiLM, MultiScaleDiffGraphNet, DualPathSpectralGraphNet,
    BioStateSpaceGraphNet, BioHeteroCausalGraphNet, HeteroGraphResNetFiLM,
    HeteroGraphDualPathNet, HeteroGraphChannelHyperFNO,
    DynamicReactionFluxOperator, SpectralTemporalHybridOperator,
    PhysicsConstrainedBiochemicalOperator,
    BioMechanisticOperator, BioMechFull,
    BioMechEventOnly, BioMechTitrationOnly, BioMechPathwayOnly,
    BioRefiner, BioRefinerCurvature,
    BioRefinerPyramid, BioRefinerGraph, BioRefinerPlus,
    BioRefinerBifurcate, BioRefinerGraphFlux, BioRefinerMechanistic,
    BioMassActionRefiner,
    BioRefinerAttractor, BioRefinerCausalReach, BioRefinerAttractorPlus,
    BioRefinerCausalBase, BioRefinerCausalCurvature,
    BioRefinerCausalPyramid, BioRefinerCausalGraph,
    BioRefinerCausalPlus, BioRefinerCausal,
    BioRefinerDynamicCausalPlus, BioRefinerLatchCausalPlus,
    BioRefinerDynamicPlus, BioEventHybridRefiner,
    HANO, HANO_NoEvent, HANO_NoFlux, HANO_NoGraph, HANO_Refine, HANO_Dual
)

from budding_yeast_v2.utils.losses import LOSS_REGISTRY, get_curriculum_weights
from budding_yeast_v2.utils.metrics import relative_l2_loss


MODEL_REGISTRY = {
    **BASELINE_REGISTRY,
    'cross_fno': NIPSYeastFNO,
    'ultimate_fno': UltimateFNO,
    'cross_wno': NIPSYeastWNO,
    'cross_lno': NIPSYeastLNO,
    'hyper_fno': HyperFNO,
    'channel_hyper_fno': ChannelHyperFNO,
    'hp_fno': hpYeastFNO,
    'graph_resnet_film': GraphResNetFiLM,
    'gated_graph_resnet_film': GatedGraphResNetFiLM,
    'multiscale_diffgraph': MultiScaleDiffGraphNet,
    'dualpath_spectralgraph': DualPathSpectralGraphNet,
    'bio_statespace_graph': BioStateSpaceGraphNet,
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
    'biom_mass_action_refiner': BioMassActionRefiner,
    'biom_mass_action': BioMassActionRefiner,
    'biom_refiner_attractor': BioRefinerAttractor,
    'bio_refiner_attractor': BioRefinerAttractor,
    'biom_refiner_causal_reach': BioRefinerCausalReach,
    'bio_refiner_causal_reach': BioRefinerCausalReach,
    'biom_refiner_attractor_plus': BioRefinerAttractorPlus,
    'bio_refiner_attractor_plus': BioRefinerAttractorPlus,
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


SINGLE_VARIABLE_GRAPH_MODELS = frozenset({
    'ultimate_fno', 'cross_fno', 'cross_wno', 'cross_lno', 'channel_hyper_fno',
    'graph_resnet_film', 'gated_graph_resnet_film', 'multiscale_diffgraph',
    'dualpath_spectralgraph', 'bio_statespace_graph',
    'biom_refiner_graph', 'biom_refiner_graph_attn', 'biom_refiner_plus',
    'biom_refiner_bifurcate', 'biom_refiner_graph_flux', 'biom_refiner_mechanistic',
    'biom_mass_action_refiner', 'biom_mass_action',
    'biom_refiner_attractor', 'bio_refiner_attractor',
    'biom_refiner_causal_reach', 'bio_refiner_causal_reach',
    'biom_refiner_attractor_plus', 'bio_refiner_attractor_plus',
    'biom_refiner_causal_graph', 'biom_refiner_causal_graph_attn',
    'biom_refiner_causal_plus', 'biom_refiner_causal',
    'biom_refiner_dynamic_causal_plus', 'biom_refiner_dynamic_causal',
    'biom_refiner_latch_causal_plus', 'biom_refiner_latch_causal',
    'biom_refiner_dynamic_plus', 'biom_refiner_event_hybrid', 'bio_event_hybrid_refiner',
})

# These architectures hold source-derived relation overlays + untyped support.
RELATION_GRAPH_MODELS = frozenset({
    'bio_hetero_causal_graph', 'hetero_graph_resnet_film', 'hetero_graph_dualpath',
    'hetero_graph_channel_hyper_fno', 'dyrfo', 'dynamic_reaction_flux_operator',
    'pc_bio_operator', 'physics_constrained_biochemical_operator',
    'biom_refiner_graph_flux', 'biom_refiner_mechanistic',
    'hano', 'hano_no_event', 'hano_no_flux', 'hano_refine', 'hano_dual',
})
MODULE_DIRECTION_REPAIRED_MODELS = frozenset({'biom_refiner_dynamic_plus'})

# Standard baselines have no Budding-specific graph or event prior.
FISSION_MODELS = frozenset(BASELINE_REGISTRY) | frozenset({
    'hyper_fno','cross_fno','cross_wno','cross_lno','channel_hyper_fno',
    'graph_resnet_film','gated_graph_resnet_film','multiscale_diffgraph','dualpath_spectralgraph',
    'bio_statespace_graph','ultimate_fno','biom_refiner_plus',
})


class YeastLitModule(pl.LightningModule):
    def __init__(
        self,
        model_name='fno',
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
        norm_type='log1p',
        graph_view='hybrid',
        graph_load_policy='current',
        adj_matrix_path=None,
        seq_len=500,
        wavelet_level=3,
        sensitivity_weight=1.0,
        system_id=None,
        min_lr=1e-6,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.system_id=system_id or ('fission' if (num_vars,param_dim)==(16,58) else 'budding')
        if system_id is not None or (num_vars,param_dim) in ((16,58),(39,141)):
            from budding_yeast_v2.data.systems.registry import resolve_system
            resolve_system(self.system_id,num_vars=num_vars,param_dim=param_dim)
        self.hparams.system_id=self.system_id
        if self.system_id=='fission' and model_name not in FISSION_MODELS:
            raise ValueError(f'{model_name} embeds Budding-specific biology; use a supported Fission architecture such as fno, cross_fno or graph_resnet_film')

        if model_name in SINGLE_VARIABLE_GRAPH_MODELS:
            if causal_matrix is not None and (adj_matrix is not None or adj_matrix_path is not None):
                raise ValueError('Specify only one variable graph override')
            if causal_matrix is not None and model_name != 'ultimate_fno':
                raise ValueError('causal_matrix is supported only by ultimate_fno')
            graph = resolve_variable_graph(
                causal_matrix if model_name == 'ultimate_fno' and causal_matrix is not None else adj_matrix,
                num_vars, view=graph_view, path=adj_matrix_path,system_id=self.system_id,
            )
            adj_matrix = graph
            causal_matrix = graph if model_name == 'ultimate_fno' else causal_matrix
            # Store override contents, not a fragile external file dependency.
            self.hparams.adj_matrix_path = None
            if graph.origin == 2:
                self.hparams.adj_matrix = graph.matrix.clone() if model_name != 'ultimate_fno' else None
                self.hparams.causal_matrix = graph.matrix.clone() if model_name == 'ultimate_fno' else None
        elif adj_matrix is not None or adj_matrix_path is not None or causal_matrix is not None:
            raise ValueError(f'{model_name} does not accept a single variable graph override')

        if model_name not in MODEL_REGISTRY:
            raise ValueError(f'Unknown or removed model {model_name!r}; standard baselines: {tuple(BASELINE_REGISTRY)}')
        ModelClass = MODEL_REGISTRY[model_name]

        if model_name in BASELINE_REGISTRY:
            self.model = build_baseline(model_name, num_vars=num_vars, param_dim=param_dim,
                width=width, n_blocks=n_blocks, modes=modes, wavelet=wavelet,
                seq_len=seq_len, level=wavelet_level,
                ode_method=ode_method, ode_rtol=ode_rtol, ode_atol=ode_atol)
        elif model_name == 'ultimate_fno':
            self.model = ModelClass(causal_matrix=causal_matrix, param_dim=param_dim, modes=modes)
        elif model_name == 'cross_fno':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, modes=modes,
                                    width=width, n_blocks=n_blocks, adj_matrix=adj_matrix)
        elif model_name == 'cross_wno':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, width=width,
                                    n_blocks=n_blocks, adj_matrix=adj_matrix, wavelet=wavelet)
        elif model_name == 'cross_lno':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, modes=modes,
                                    width=width, n_blocks=n_blocks, adj_matrix=adj_matrix)
        elif model_name == 'hyper_fno':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, modes=modes,
                                    width=width, n_layers=n_blocks, d_lambda=d_lambda)
        elif model_name == 'channel_hyper_fno':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, modes=modes,
                                    width=width, n_layers=n_blocks, d_lambda=d_lambda,
                                    adj_matrix=adj_matrix)
        elif model_name in ('graph_resnet_film', 'gated_graph_resnet_film', 'multiscale_diffgraph', 'bio_statespace_graph'):
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim,
                                    width=width, n_blocks=n_blocks, d_lambda=d_lambda,
                                    adj_matrix=adj_matrix)
        elif model_name == 'dualpath_spectralgraph':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim,
                                    width=width, modes=modes, n_blocks=n_blocks, d_lambda=d_lambda,
                                    adj_matrix=adj_matrix)
        elif model_name == 'hp_fno':
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim, modes=modes, width=width)
        elif model_name in ('bio_hetero_causal_graph', 'hetero_graph_resnet_film'):
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim,
                                    width=width, n_blocks=n_blocks, d_lambda=d_lambda)
        elif model_name in ('hetero_graph_dualpath', 'hetero_graph_channel_hyper_fno'):
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim,
                                    modes=modes, width=width, n_blocks=n_blocks, d_lambda=d_lambda)
        elif model_name in ('dyrfo', 'dynamic_reaction_flux_operator', 'pc_bio_operator', 'physics_constrained_biochemical_operator'):
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim,
                                    width=width, n_blocks=n_blocks)
        elif model_name in ('st_hyperno', 'spectral_temporal_hybrid_operator', 'biom_full', 'biom_net', 'bio_mechanistic_operator', 'biom_event', 'biom_titration', 'biom_pathway', 'biom_refiner', 'bio_refiner', 'biom_refiner_curvature', 'biom_refiner_curv', 'biom_refiner_pyramid', 'biom_refiner_pyr', 'biom_refiner_causal_base', 'biom_refiner_causal_curvature', 'biom_refiner_causal_curv', 'biom_refiner_causal_pyramid', 'biom_refiner_causal_pyr'):
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim,
                                    modes=modes, width=width, n_blocks=n_blocks)
        elif model_name in ('biom_refiner_graph', 'biom_refiner_graph_attn', 'biom_refiner_plus', 'biom_refiner_bifurcate', 'biom_refiner_graph_flux', 'biom_refiner_mechanistic', 'biom_mass_action_refiner', 'biom_mass_action', 'biom_refiner_attractor', 'bio_refiner_attractor', 'biom_refiner_causal_reach', 'bio_refiner_causal_reach', 'biom_refiner_attractor_plus', 'bio_refiner_attractor_plus', 'biom_refiner_causal_graph', 'biom_refiner_causal_graph_attn', 'biom_refiner_causal_plus', 'biom_refiner_causal', 'biom_refiner_dynamic_causal_plus', 'biom_refiner_dynamic_causal', 'biom_refiner_latch_causal_plus', 'biom_refiner_latch_causal', 'biom_refiner_dynamic_plus', 'biom_refiner_event_hybrid', 'bio_event_hybrid_refiner'):
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim,
                                    modes=modes, width=width, n_blocks=n_blocks,
                                    adj_matrix=adj_matrix)
        elif model_name in ('hano', 'hano_no_event', 'hano_no_flux', 'hano_no_graph', 'hano_refine', 'hano_dual'):
            self.model = ModelClass(num_vars=num_vars, param_dim=param_dim,
                                    width=width, n_blocks=n_blocks)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        configure_relation_view(self.model, graph_view)
        set_graph_load_policy(self.model, graph_load_policy)
        loss_class = LOSS_REGISTRY[loss_type]
        if loss_type == 'sc_sensitivity':
            if model_name != 'sc_fno':
                raise ValueError('sc_sensitivity is the SC-FNO training loss; use model=sc_fno')
            self.criterion = loss_class(sensitivity_weight=sensitivity_weight)
        elif loss_type in ('mse_only', 'mse_smooth', 'h1', 'rel_l2', 'hybrid_mse_rel', 'gradnorm_hybrid',
                         'phase_derivative', 'soft_dtw', 'hybrid_phase_deriv', 'hybrid_soft_dtw',
                         'triple_dynamic', 'hybrid_phase_dtw'):
            self.criterion = loss_class(num_vars=num_vars) if loss_type=='gradnorm_hybrid' else loss_class()
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
                't_max': 300.0 if self.system_id=='fission' else 210.0,
                'norm_type':norm_type,
                's_c':torch.ones(1,num_vars,1) if norm_type=='asinh_scale' else None,
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
            self.criterion = loss_class(torch.zeros(1, num_vars, 1), torch.ones(1, num_vars, 1))
        self.criterion.system_id=self.system_id

        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.adj_matrix = adj_matrix

        # 归一化统计量与反归一化状态 (非持久化 buffer，训练时从 datamodule 注入)
        self.register_buffer('mean_y', torch.zeros(1, num_vars, 1), persistent=False)
        self.register_buffer('std_y', torch.ones(1, num_vars, 1), persistent=False)
        self.register_buffer('s_c', torch.ones(1, num_vars, 1), persistent=False)
        self.register_buffer('p_mean', torch.zeros(1, param_dim), persistent=False)
        self.register_buffer('p_std', torch.ones(1, param_dim), persistent=False)
        self.norm_type = norm_type
        self._preprocessing_state = None
        self._legacy_fit_scope = None
        self._stats_bound = False

    def setup(self, stage=None):
        # HANO 等结构保持模型需要在真实空间完成通量积分/事件跳变,
        # 注入 datamodule 归一化统计量 (persistent buffer, 随 checkpoint 持久化)
        self._inject_model_stats()

    def graph_report(self):
        fixed = fixed_graph_report(self.model)
        relations = [r for r in fixed if r['kind'] == 'relations']
        modules = [r for r in fixed if r['kind'] == 'module']
        return {
            'variable_graphs': variable_graph_report(self.model),
            'relation_graphs': relations,
            'module_graphs': modules,
            'legacy_six_relation_branch': any(r['origin'] == 'legacy_unknown' for r in relations),
            'module_graph_migrated': bool(modules) and all(r['origin'] != 'legacy_unknown' for r in modules),
        }

    def set_parameter_stats(self, p_mean=None, p_std=None, norm_type=None):
        """Bind DataModule parameter statistics for causal physical WT features."""
        setter = getattr(self.model, 'set_parameter_stats', None)
        if setter is not None:
            setter(p_mean, p_std, self.norm_type if norm_type is None else norm_type)

    def _inject_model_stats(self):
        trainer = getattr(self, '_trainer', None)
        dm = getattr(trainer, 'datamodule', None) if trainer is not None else None
        if dm is None or not hasattr(dm, 'mean_y') or dm.mean_y is None:
            return
        self.bind_datamodule_stats(dm)

    def bind_datamodule_stats(self, dm):
        """Validate a saved contract before binding every physical decoder."""
        if getattr(dm,'system_id',self.system_id)!=self.system_id:
            raise ValueError('Checkpoint/model biological system does not match the dataset')
        state_fn = getattr(dm, 'preprocessing_state', None)
        state = state_fn() if callable(state_fn) else None
        if (self._legacy_fit_scope is not None and state is not None
                and state['fit_scope'] != self._legacy_fit_scope):
            raise ValueError('Legacy checkpoint normalization fit_scope mismatch: '
                             f'construct its DataModule with fit_scope={self._legacy_fit_scope!r}; '
                             'do not apply train-only statistics to legacy normalized weights')
        if self._preprocessing_state is not None and state is not None:
            assert_compatible(self._preprocessing_state, state)
        for key in ('mean_y', 'std_y', 'p_mean', 'p_std'):
            getattr(self, key).copy_(getattr(dm, key).to(self.device))
        self.norm_type = getattr(dm, 'norm_type', getattr(self, 'norm_type', 'log1p'))
        self.hparams.norm_type = self.norm_type
        self.set_parameter_stats(dm.p_mean, dm.p_std, self.norm_type)
        if getattr(dm, 's_c', None) is not None:
            self.s_c.copy_(dm.s_c.to(self.device))
        elif self.norm_type == 'asinh_scale':
            raise ValueError('asinh_scale requires DataModule s_c')
        else:
            self.s_c.fill_(1.0)
        if hasattr(self.model, 'set_stats'):
            self.model.set_stats(dm.mean_y, dm.std_y, dm.p_mean, dm.p_std, float(dm.t_max),
                                 norm_type=self.norm_type, s_c=getattr(dm, 's_c', None))
        self._bind_criterion_stats(float(dm.t_max))
        if state is not None:
            self._preprocessing_state = state
        self._stats_bound = True

    def _bind_criterion_stats(self, t_max):
        if hasattr(self.criterion, 'set_stats'):
            self.criterion.set_stats(self.mean_y, self.std_y, self.p_mean, self.p_std, t_max,
                                     norm_type=self.norm_type, s_c=self.s_c if self.norm_type == 'asinh_scale' else None)
        elif hasattr(self.criterion, 'bound_norm'):
            self.criterion.bound_norm = -self.mean_y / self.std_y

    def on_save_checkpoint(self, checkpoint):
        self._inject_model_stats()
        if self._preprocessing_state is not None:
            checkpoint['preprocessing_contract'] = copy.deepcopy(self._preprocessing_state)

    def on_load_checkpoint(self, checkpoint):
        # Baseline names now refer to standard architectures; old modified
        # checkpoints are intentionally not reconstructed under these names.
        hp = checkpoint.get('hyper_parameters', {})
        recorded = (checkpoint.get('preprocessing_contract') is not None or hp.get('norm_type') is not None
                    or checkpoint.get('datamodule_hyper_parameters', {}).get('norm_type') is not None)
        self.norm_type = resolve_checkpoint_norm(checkpoint) if recorded else self.norm_type
        self.hparams.norm_type = self.norm_type
        state = checkpoint.get('preprocessing_contract')
        if state is None:
            self._legacy_fit_scope = resolve_checkpoint_fit_scope(checkpoint)
            warnings.warn('Legacy checkpoint has no full preprocessing contract; '
                          'rebuild statistics from its original dataset with '
                          f'fit_scope={self._legacy_fit_scope!r} before physical inference.')
            self._inject_model_stats()
            return
        self._legacy_fit_scope = None
        state = validate_preprocessing_state(state, self.hparams.num_vars, self.hparams.param_dim)
        for prefix in ('model.', 'criterion.'):
            for key in ('mean_y', 'std_y', 'p_mean', 'p_std'):
                saved = checkpoint.get('state_dict', {}).get(prefix + key)
                if saved is not None and not torch.equal(saved.detach().cpu(), state[key]):
                    raise ValueError(f'Checkpoint buffers contradict preprocessing contract: {prefix}{key}')
        self._preprocessing_state = state
        for key in ('mean_y', 'std_y', 'p_mean', 'p_std'):
            getattr(self, key).copy_(state[key].to(self.device))
        self.s_c.copy_(torch.ones_like(self.s_c) if state.get('s_c') is None else state['s_c'].to(self.device))
        self.set_parameter_stats(self.p_mean, self.p_std, self.norm_type)
        if hasattr(self.model, 'set_stats'):
            self.model.set_stats(self.mean_y, self.std_y, self.p_mean, self.p_std,
                                 state['t_max'], norm_type=self.norm_type, s_c=state.get('s_c'))
        self._bind_criterion_stats(state['t_max'])
        self._stats_bound = True
        self._inject_model_stats()

    def denormalize_y(self, y_norm):
        """
        将归一化张量 y_norm (B, V, T) 反归一化到真实物理浓度空间。
        优先使用 trainer.datamodule.denormalize_y，兜底使用自身缓存的 mean_y / std_y / norm_type。
        """
        trainer = getattr(self, '_trainer', None)
        dm = getattr(trainer, 'datamodule', None) if trainer is not None else None
        if dm is not None and hasattr(dm, 'denormalize_y'):
            return dm.denormalize_y(y_norm, device=y_norm.device)

        if not self._stats_bound:
            raise RuntimeError('Physical denormalization requires bound/checkpoint preprocessing statistics')
        return decode_y(y_norm, self.mean_y, self.std_y, self.norm_type, self.s_c)

    def on_train_start(self):
        if self.hparams.model_name == 'sc_fno':
            if self.hparams.loss_type != 'sc_sensitivity':
                raise ValueError('SC-FNO requires loss_type=sc_sensitivity and reference sensitivity targets')
            if self.hparams.rollout_type != 'one_time':
                raise ValueError('SC-FNO sensitivity targets currently use the one_time trajectory contract')
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
                    'norm_type': getattr(dm, 'norm_type', 'log1p'),
                    's_c': getattr(dm, 's_c', None),
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

        self.criterion.system_id=self.system_id

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

    def _autoregressive_rollout(self, x, p, *, return_coarse=False, return_pattern=False, return_aux=False):
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

        # Use the identical rollout for training and inference. Auxiliary
        # coarse/event heads supervise the same visited states, not an unrelated
        # one-shot trajectory. Lists avoid mutable autograd views.
        states = [x[:, :, 0, 0]]
        coarse_states = [states[0]]
        pattern_logits = None
        auxiliary = {}

        # 逐步滚动：从第 0 步预测第 1 步，第 1 步预测第 2 步，...
        for k in range(T - 1):
            current = states[-1].unsqueeze(-1).expand(-1, -1, T)
            temp_input = torch.stack((current, x[:, :, 1, :]), dim=2)
            if return_aux:
                out, aux = self.model(temp_input, p, return_aux=True)
                for key, value in aux.items():
                    if isinstance(value, torch.Tensor) and value.ndim == 3 and value.shape[-1] == T:
                        if key not in auxiliary:
                            initial = states[0] if key == 'physics_out' else torch.zeros_like(value[:, :, 0])
                            auxiliary[key] = [initial]
                        auxiliary[key].append(value[:, :, 1])
            elif return_coarse:
                if return_pattern:
                    out, coarse, logits = self.model(temp_input, p, return_coarse=True, return_pattern=True)
                    if k == 0:
                        pattern_logits = logits
                else:
                    out, coarse = self.model(temp_input, p, return_coarse=True)
                coarse_states.append(coarse[:, :, 1])
            else:
                out = self.model(temp_input, p)
            states.append(out[:, :, 1])
        trajectory = torch.stack(states, dim=-1)
        if return_aux:
            return trajectory, {key: torch.stack(value, dim=-1) for key, value in auxiliary.items()}
        if return_coarse:
            coarse = torch.stack(coarse_states, dim=-1)
            return (trajectory, coarse, pattern_logits) if return_pattern else (trajectory, coarse)
        return trajectory

    def training_step(self, batch, batch_idx):
        sensitivity_batch = batch if isinstance(batch, dict) else None
        if sensitivity_batch is not None:
            x, p, y = (batch[key] for key in ('x', 'p', 'y'))
            pat_id = batch.get('pattern')
        elif len(batch) == 4:
            x, p, y, pat_id = batch
        else:
            x, p, y = batch
            pat_id = None

        curr_ep = getattr(self, 'current_epoch', 0)
        lam_pen, lam_sm = get_curriculum_weights(curr_ep, self.max_epochs)

        weights = torch.ones(len(x), dtype=torch.float32, device=self.device)

        if self.hparams.model_name.startswith('hano') and self.training:
            if self.hparams.rollout_type == 'autoregressive':
                out_full, aux = self._autoregressive_rollout(x, p, return_aux=True)
            else:
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
            forward = self._autoregressive_rollout if self.hparams.rollout_type == 'autoregressive' else self.model
            if hasattr(self.model, 'pattern_head'):
                out_full, out_coarse, pattern_logits = forward(x, p, return_coarse=True, return_pattern=True)
            else:
                out_full, out_coarse = forward(x, p, return_coarse=True)

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
            elif self.hparams.loss_type == 'gradnorm_hybrid':
                loss_final = self.criterion(out_full, y, weights, lam_pen, lam_sm, update_weights=True)
                loss_coarse = self.criterion(out_coarse, y, weights, lam_pen, lam_sm, update_weights=False)
                loss = loss_final + 0.3 * loss_coarse
            else:
                loss_final = self.criterion(out_full, y, weights, lam_pen, lam_sm)
                loss_coarse = self.criterion(out_coarse, y, weights, lam_pen, lam_sm)
                loss = loss_final + 0.3 * loss_coarse
        else:
            out_full = self.forward_ic_time(x, p)

            if self.hparams.model_name == 'sc_fno':
                if self.hparams.loss_type != 'sc_sensitivity' or sensitivity_batch is None:
                    raise ValueError('SC-FNO training requires sc_sensitivity loss and sensitivity targets')
                loss = self.criterion(out_full, y) + self.criterion.sensitivity_term(
                    self.model, x, p, batch['sensitivity_targets'], batch['sensitivity_indices'])
            elif self.hparams.loss_type in ('pinn_residual', 'pino', 'hybrid_pino_colloc', 'hybrid_all'):
                self.criterion.set_epoch(curr_ep)
                loss = self.criterion(out_full, y, p, weights, lam_pen, lam_sm)
                if hasattr(self.criterion, '_last_loss_components'):
                    for k, v in self.criterion._last_loss_components.items():
                        self.log(f'train_{k}', v, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            elif self.hparams.loss_type == 'hybrid_bifurcate':
                loss = self.criterion(out_full, y, pattern_targets=pat_id)
            else:
                loss = self.criterion(out_full, y, weights, lam_pen, lam_sm)

        # 记录准实时自适应损失诊断指标 (如 GradNorm 权重)
        if hasattr(self.criterion, '_last_metrics') and self.criterion._last_metrics:
            for k, v in self.criterion._last_metrics.items():
                self.log(f'train_{k}', v, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x_v, p_v, y_v = batch[:3]
        pred = self.forward_ic_time(x_v, p_v)
        target = y_v
        mse_val = F.mse_loss(pred, target)

        if dataloader_idx == 0:
            self.log('val_loss_synth', mse_val, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False, sync_dist=True)
        elif dataloader_idx == 1:
            self.log('test_loss_synth', mse_val, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False, sync_dist=True)
        elif dataloader_idx == 2:
            self.log('test_loss_real', mse_val, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False, sync_dist=True)
            # 计算真实物理空间下的 Real Mutants 损失 (使用 metrics.py 同款 Relative L2 Loss, eps=0.1 物理基底截断保护)
            pred_real = self.denormalize_y(pred)
            target_real = self.denormalize_y(target)
            rel_l2_real = relative_l2_loss(pred_real, target_real, eps=0.1)
            # 1. 新标准名称 (显式标注反归一化物理空间与相对 L2 误差)
            self.log('test_real_denorm_rel_l2', rel_l2_real, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False, sync_dist=True)
            # 2. 兼容历史别名
            self.log('test_loss_real_denorm', rel_l2_real, on_step=False, on_epoch=True, add_dataloader_idx=False, sync_dist=True)

        return mse_val

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        warmup_epochs = self.hparams.get('warmup_epochs', 10)
        max_epochs = self.max_epochs
        eta_min = float(self.hparams.get('min_lr', 1e-6))
        if not math.isfinite(self.lr) or self.lr <= 0 or not math.isfinite(eta_min) or not 0 <= eta_min <= self.lr:
            raise ValueError('min_lr must be finite and between zero and the peak learning rate')

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # 线性 warmup: 0 → 1
                return epoch / max(warmup_epochs, 1)
            # Cosine annealing: 1 → eta_min/lr
            progress = min(1., max(0., (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)))
            return eta_min / self.lr + 0.5 * (1.0 - eta_min / self.lr) * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            },
        }
