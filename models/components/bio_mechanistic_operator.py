#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioMechanisticOperator (BioMechNet)
===================================
A unified biophysical neural operator integrating:
1. NeuralEventTrigger: Mass-gated continuous first-passage time integrator (fixes G1 Start delay).
2. BiochemicalTitrationActivation: Differentiable quadratic equilibrium solver for bimolecular binding (fixes stoichiometric titration kinks).
3. PathwayBypassMoE: Adaptive biochemical bypass routing under gene knockouts.
Built on top of the champion ST-HyperNO and ResNetFiLM (width=128) backbones.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """1D Continuous Spectral Convolution with AMP Float32 stability."""
    def __init__(self, in_channels, out_channels, modes1=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        orig_dtype = x.dtype
        x_f32 = x.float()
        B, C, T = x_f32.shape

        x_ft = torch.fft.rfft(x_f32, dim=-1)
        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1], device=x.device, dtype=torch.cfloat)

        modes = min(self.modes1, x_ft.shape[-1])
        out_ft[:, :, :modes] = self.compl_mul1d(x_ft[:, :, :modes], self.weights1[:, :, :modes])

        x_out = torch.fft.irfft(out_ft, n=T, dim=-1)
        return x_out.to(orig_dtype)


# =========================================================================
# 1. 机制模块 1: NeuralEventTrigger (临界质量事件积分器)
# =========================================================================
class NeuralEventTrigger(nn.Module):
    """
    Continuous Mass-Gated Neural Event Integrator.
    Tracks cell mass growth MASS(t) to predict first-passage time across critical threshold θ(p).
    Generates time-dependent continuous event mask E(t) = Sigmoid((MASS(t) - θ(p)) / τ).
    """
    def __init__(self, width=128, param_dim=141):
        super().__init__()
        self.width = width
        # Parameter-dependent threshold θ(p) and sharpness τ(p) generator
        self.threshold_net = nn.Sequential(
            nn.Linear(param_dim, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 2) # [theta_offset, tau_raw]
        )
        self.event_proj = nn.Sequential(
            nn.Conv1d(2, width // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(width // 4, width, kernel_size=1)
        )

    def forward(self, x_in, p):
        """
        x_in: (B, 78, T) or (B, 39, 2, T)
        p: (B, 141)
        Returns: event_emb (B, width, T)
        """
        if x_in.dim() == 4:
            mass = x_in[:, 0, 0, :] # (B, T)
            time_grid = x_in[:, 0, 1, :] # (B, T)
        else:
            mass = x_in[:, 0, :] # (B, T)
            time_grid = x_in[:, 39, :] if x_in.shape[1] > 39 else torch.linspace(0, 1, x_in.shape[-1], device=x_in.device).expand(x_in.shape[0], -1)

        # Generate adaptive threshold and timescale
        theta_params = self.threshold_net(p) # (B, 2)
        theta = theta_params[:, 0:1] # (B, 1)
        tau = F.softplus(theta_params[:, 1:2]) + 0.05 # (B, 1)

        # Continuous differentiable event trigger
        diff = (mass - theta) / tau # (B, T)
        event_mask = torch.sigmoid(diff) # (B, T)

        # Combined event feature: [event_mask, d(event)/dt approx]
        event_input = torch.stack([event_mask, time_grid], dim=1) # (B, 2, T)
        event_emb = self.event_proj(event_input) # (B, width, T)
        return event_emb


# =========================================================================
# 2. 机制模块 2: BiochemicalTitrationActivation (生化滴定与结合激活算子)
# =========================================================================
class BiochemicalTitrationActivation(nn.Module):
    """
    Differentiable Biochemical Stoichiometric Titration & Reversible Complex Equilibrium Layer.
    Solves closed-form quadratic equilibrium of A + B <-> AB with dissociation constant Kd:
    [A]_free = ((A - B - Kd) + sqrt((A - B - Kd)^2 + 4 * A * Kd)) / 2
    """
    def __init__(self, width=128, param_dim=141):
        super().__init__()
        self.width = width
        self.half_w = width // 2
        
        self.kd_net = nn.Sequential(
            nn.Linear(param_dim, self.half_w),
            nn.GELU(),
            nn.Linear(self.half_w, self.half_w),
            nn.Softplus()
        )
        self.blend = nn.Parameter(torch.ones(width) * 0.5)

    def forward(self, h, p):
        """
        h: (B, width, T)
        p: (B, param_dim)
        """
        B, W, T = h.shape
        # Split into activator and inhibitor channels
        A = F.softplus(h[:, :self.half_w, :]) # (B, W/2, T)
        B_conc = F.softplus(h[:, self.half_w:, :]) # (B, W/2, T)

        # Dissociation constant Kd(p)
        Kd = (self.kd_net(p).unsqueeze(-1) + 1e-4).expand(-1, -1, T) # (B, W/2, T)

        # Exact algebraic equilibrium solution
        delta = A - B_conc - Kd
        discriminant = torch.clamp(delta ** 2 + 4.0 * A * Kd, min=1e-8)
        A_free = (delta + torch.sqrt(discriminant)) * 0.5 # (B, W/2, T)

        # Symmetric solution for B_free
        delta_B = B_conc - A - Kd
        disc_B = torch.clamp(delta_B ** 2 + 4.0 * B_conc * Kd, min=1e-8)
        B_free = (delta_B + torch.sqrt(disc_B)) * 0.5 # (B, W/2, T)

        titrated = torch.cat([A_free, B_free], dim=1) # (B, width, T)

        # Soft residual blend with standard GELU
        alpha = torch.sigmoid(self.blend.view(1, -1, 1))
        out = alpha * titrated + (1.0 - alpha) * F.gelu(h)
        return out


# =========================================================================
# 3. 机制模块 3: PathwayBypassMoE (自适应生化旁路路由网络)
# =========================================================================
class PathwayBypassMoE(nn.Module):
    """
    Adaptive Biochemical Pathway Bypass Mixture-of-Experts.
    Detects severe pathway blockages (gene knockout p_i = 0) and dynamically routes
    feature representations to alternative bypass compensation pathways.
    """
    def __init__(self, width=128, param_dim=141, dilation=1):
        super().__init__()
        self.width = width

        # Router Network
        self.router = nn.Sequential(
            nn.Linear(param_dim, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 2), # [w_primary, w_bypass]
            nn.Softmax(dim=-1)
        )

        # 1. Primary Pathway Stream (Standard High-flux propagation)
        self.primary_conv = nn.Sequential(
            nn.GroupNorm(1, width),
            nn.Conv1d(width, width, kernel_size=5, padding=2 * dilation, dilation=dilation),
            nn.GELU()
        )

        # 2. Alternative Bypass Stream (Wide-receptive bypass compensation)
        bypass_dilation = max(1, dilation * 2)
        self.bypass_conv = nn.Sequential(
            nn.GroupNorm(1, width),
            nn.Conv1d(width, width, kernel_size=3, padding=bypass_dilation, dilation=bypass_dilation),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=1)
        )

    def forward(self, h, p):
        """
        h: (B, width, T)
        p: (B, param_dim)
        """
        B, W, T = h.shape
        routing_weights = self.router(p) # (B, 2)
        w_prim = routing_weights[:, 0].view(B, 1, 1) # (B, 1, 1)
        w_byp = routing_weights[:, 1].view(B, 1, 1)

        h_prim = self.primary_conv(h)
        h_byp = self.bypass_conv(h)

        out = w_prim * h_prim + w_byp * h_byp
        return out


# =========================================================================
# 4. 统一物理生化残差块 (BioMechanisticBlock)
# =========================================================================
class BioMechanisticBlock(nn.Module):
    """
    Unified Residual Block combining Spectral-Temporal streams with Titration Activation & Pathway MoE.
    """
    def __init__(self, width=128, modes=32, dilation=1, param_dim=141,
                 use_titration=True, use_pathway=True):
        super().__init__()
        self.width = width
        self.use_titration = use_titration
        self.use_pathway = use_pathway

        # 1. Global Spectral Stream (Fourier Neural Operator)
        self.spectral_conv = SpectralConv1d(width, width, modes1=modes)
        self.spectral_linear = nn.Conv1d(width, width, kernel_size=1)
        self.spectral_norm = nn.GroupNorm(1, width)

        # 2. Local Dilated Temporal Stream with Pathway MoE
        self.temp_norm1 = nn.GroupNorm(1, width)
        self.temp_conv1 = nn.Conv1d(width, width, kernel_size=5, padding=2 * dilation, dilation=dilation)
        
        if self.use_pathway:
            self.pathway_moe = PathwayBypassMoE(width=width, param_dim=param_dim, dilation=dilation)
        else:
            self.pathway_moe = None

        self.temp_norm2 = nn.GroupNorm(1, width)
        self.temp_conv2 = nn.Conv1d(width, width, kernel_size=5, padding=2 * dilation, dilation=dilation)

        # 3. Biochemical Titration Activation
        if self.use_titration:
            self.titration_act = BiochemicalTitrationActivation(width=width, param_dim=param_dim)
        else:
            self.titration_act = None

        # 4. FiLM Parameter Conditioning
        self.film = nn.Sequential(
            nn.Linear(param_dim, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width * 2)
        )
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

        # 5. Cross-Domain Gated Fusion
        self.gate_net = nn.Sequential(
            nn.Conv1d(width * 2, width, kernel_size=1),
            nn.Sigmoid()
        )
        self.act = nn.GELU()

    def forward(self, h_spec, h_temp, p):
        res_spec = h_spec
        res_temp = h_temp

        # A. Spectral Stream Update
        s_out = self.spectral_conv(h_spec) + self.spectral_linear(h_spec)
        s_out = self.act(self.spectral_norm(s_out))

        # B. Temporal Stream Update with FiLM & Pathway MoE
        t_h = self.temp_norm1(h_temp)
        film_val = self.film(p).unsqueeze(-1)
        gamma, beta = torch.chunk(film_val, 2, dim=1)
        t_h = t_h * (1.0 + torch.tanh(gamma)) + beta

        if self.titration_act is not None:
            t_h = self.titration_act(self.temp_conv1(t_h), p)
        else:
            t_h = self.act(self.temp_conv1(t_h))

        if self.pathway_moe is not None:
            t_h = self.pathway_moe(t_h, p)

        t_h = self.temp_norm2(t_h)
        t_out = self.temp_conv2(t_h)

        # C. Cross-Domain Gated Interaction
        cat_feat = torch.cat([s_out, t_out], dim=1)
        gate = self.gate_net(cat_feat)

        h_spec_next = self.act(res_spec + s_out * gate + t_out * (1.0 - gate))
        h_temp_next = self.act(res_temp + t_out * (1.0 - gate) + s_out * gate)

        return h_spec_next, h_temp_next


# =========================================================================
# 5. 旗舰架构主类 (BioMechanisticOperator)
# =========================================================================
class BioMechanisticOperator(nn.Module):
    """
    Flagship BioMechanisticOperator (BioMechNet) Architecture
    """
    def __init__(
        self,
        num_vars=39,
        param_dim=141,
        modes=32,
        width=128,
        n_blocks=4,
        use_event=True,
        use_titration=True,
        use_pathway=True,
        **kwargs
    ):
        super().__init__()
        self.num_vars = num_vars
        self.param_dim = param_dim
        self.modes = modes
        self.width = width
        self.n_blocks = n_blocks
        self.use_event = use_event
        self.use_titration = use_titration
        self.use_pathway = use_pathway

        # 1. Stem Lifting
        in_channels = num_vars * 2
        self.stem_spec = nn.Conv1d(in_channels, width, kernel_size=1)
        self.stem_temp = nn.Conv1d(in_channels, width, kernel_size=5, padding=2)

        # 2. Neural Event Trigger (Mass Integrator)
        if self.use_event:
            self.event_trigger = NeuralEventTrigger(width=width, param_dim=param_dim)
        else:
            self.event_trigger = None

        # 3. Multi-layer BioMechanistic Blocks
        dilations = [2 ** (i % 4) for i in range(n_blocks)]
        self.blocks = nn.ModuleList([
            BioMechanisticBlock(
                width=width,
                modes=modes,
                dilation=dilations[i],
                param_dim=param_dim,
                use_titration=use_titration,
                use_pathway=use_pathway
            )
            for i in range(n_blocks)
        ])

        # 4. Channel Self-Attention across 39 variables
        self.to_vars = nn.Linear(width, num_vars * 16)
        self.from_vars = nn.Linear(num_vars * 16, width)
        self.q_proj = nn.Linear(16, 16)
        self.k_proj = nn.Linear(16, 16)
        self.v_proj = nn.Linear(16, 16)

        # 5. Output Projection Head
        self.head = nn.Sequential(
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(width, num_vars, kernel_size=1)
        )

    def forward(self, x, p):
        """
        x: (B, 39, 2, T) or (B, 39, T)
        p: (B, 141)
        """
        if x.dim() == 4:
            B, V, F_in, T = x.shape
            x_in = x.reshape(B, V * F_in, T) # (B, 78, T)
        else:
            B, V, T = x.shape
            time_grid = torch.linspace(0, 1, T, device=x.device).view(1, 1, T).repeat(B, V, 1)
            x_in = torch.cat([x, time_grid], dim=1)

        # 1. Dual-stream stem initialization
        h_spec = self.stem_spec(x_in)
        h_temp = self.stem_temp(x_in)

        # 2. Inject Neural Event Trigger (Mass-Gated First-Passage Time)
        if self.event_trigger is not None:
            e_emb = self.event_trigger(x, p)
            h_temp = h_temp + e_emb

        # 3. Process through BioMechanistic blocks
        for block in self.blocks:
            h_spec, h_temp = block(h_spec, h_temp, p)

        # 4. Dual-domain fusion
        h = h_spec + h_temp

        # 5. Dynamic Channel Self-Attention across 39 variables
        x_T = h.permute(0, 2, 1) # (B, T, width)
        x_vars = self.to_vars(x_T).reshape(B, T, self.num_vars, 16)
        Q = self.q_proj(x_vars)
        K = self.k_proj(x_vars)
        V = self.v_proj(x_vars)
        attn = F.softmax(torch.matmul(Q, K.transpose(-1, -2)) / 4.0, dim=-1)
        out_vars = torch.matmul(attn, V).reshape(B, T, self.num_vars * 16)
        h_attn = self.from_vars(out_vars).permute(0, 2, 1)
        h = h + h_attn

        # 6. Output projection
        out = self.head(h) # (B, 39, T)
        return out


# =========================================================================
# 6. 派生消融子类（无需修改任何现有外部脚本，直接按类名调用）
# =========================================================================
class BioMechFull(BioMechanisticOperator):
    """旗舰完整版：三大物理机制全开 (Event + Titration + Pathway)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_event=True, use_titration=True, use_pathway=True, **kwargs)


class BioMechEventOnly(BioMechanisticOperator):
    """消融组 1: 仅开启临界质量事件积分 (Event Only)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_event=True, use_titration=False, use_pathway=False, **kwargs)


class BioMechTitrationOnly(BioMechanisticOperator):
    """消融组 2: 仅开启生化滴定激活算子 (Titration Only)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_event=False, use_titration=True, use_pathway=False, **kwargs)


class BioMechPathwayOnly(BioMechanisticOperator):
    """消融组 3: 仅开启自适应旁路路由 (Pathway Only)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_event=False, use_titration=False, use_pathway=True, **kwargs)
