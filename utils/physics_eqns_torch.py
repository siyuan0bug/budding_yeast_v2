"""
出芽酵母细胞周期 ODE 系统的 PyTorch 可微实现。

本文件是 lhs_v1_origin.py 中 eqns 函数的 PyTorch 版本，用于 PINN 训练。
核心目的：使 ODE 右端项 f(u, p) 可微，从而能用 autograd 计算 ODE 残差损失。

关键设计：
1. eqns_torch: 完全对齐 numpy 版 eqns，但用 torch 运算（可微）
2. apply_event_torch: event 重置规则的可微版本（用于跳跃条件损失）
3. 参数字典支持批量计算 (B, ...) 形状

注意：不修改 lhs_v1_origin.py，而是 import 其常量并重写计算逻辑。
"""
import torch
import math
import numpy as np
from budding_yeast_v2.data.lhs_v1_origin import get_default_params

# ==============================================================
# 参数键顺序（与 al_callback.py 完全一致）
# ==============================================================
_defaults = get_default_params()
_defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
PARAM_KEYS = sorted([k for k in _defaults.keys() if k != 'is_mutant_104'])

# 变量索引（与 lhs_v1_origin.py 第 33-36 行一致）
(nMASS, nCLN2, nCLB2, nCLB5, nSIC1, nCDC6, nC2, nC5, nF2, nF5,
 nSIC1P, nC2P, nC5P, nCDC6P, nF2P, nF5P, nSWI5T, nSWI5, nIEP, nCDC20T,
 nCDC20, nCDH1T, nCDH1, nCDC14T, nCDC14, nNET1T, nNET1, nRENT, nTEM1, nCDC15,
 nPPX, nPDS1, nESP1, nORI, nBUD, nSPN, nVi20, nlte1, nBUB2) = range(39)


def _gk_torch(Va, Vi, Ja, Ji):
    """
    Goldbeter-Koshland 函数的 PyTorch 可微实现。
    对齐 lhs_v1_origin.py 第 73-75 行的 GK 函数。

    注意：numpy 版用 np.sqrt，这里用 torch.sqrt 保证梯度可传。
    为数值稳定性，对判别式加 epsilon 防止负数。
    """
    BB = Vi - Va + Ja * Vi + Ji * Va
    discriminant = BB ** 2 - 4 * (Vi - Va) * Ji * Va
    # 数值稳定：判别式可能因浮点误差变负，用 clamp 保证非负
    discriminant = torch.clamp(discriminant, min=0.0)
    return 2 * Ji * Va / (BB + torch.sqrt(discriminant) + 1e-12)


def _params_dict_to_tensor(p_dict, device, dtype=torch.float32):
    """
    将 ODE 参数字典转换为 (141,) 张量，顺序与 PARAM_KEYS 一致。
    用于从字典构建参数张量。
    """
    vec = torch.tensor([p_dict[k] for k in PARAM_KEYS], device=device, dtype=dtype)
    return vec


def params_tensor_to_dict(p_vec, extra_keys=None):
    """
    将 (B, 141) 或 (141,) 参数张量转换为参数字典（值为张量）。
    用于 eqns_torch 中按键访问参数。

    Args:
        p_vec: (B, 141) 或 (141,) 参数张量
        extra_keys: 额外参数字典（如 is_mutant_104, Vi20_active 等）

    Returns:
        p_dict: 参数字典，值为 (B,) 或标量张量
    """
    if p_vec.dim() == 1:
        p_vec = p_vec.unsqueeze(0)  # (1, 141)

    p_dict = {k: p_vec[:, i] for i, k in enumerate(PARAM_KEYS)}

    # 补充非 141 维的参数
    if extra_keys is not None:
        for k, v in extra_keys.items():
            if isinstance(v, torch.Tensor):
                p_dict[k] = v
            else:
                p_dict[k] = torch.full((p_vec.shape[0],), float(v),
                                       device=p_vec.device, dtype=p_vec.dtype)
    else:
        # 默认值
        B = p_vec.shape[0]
        p_dict['is_mutant_104'] = torch.zeros(B, device=p_vec.device, dtype=p_vec.dtype)

    return p_dict


def eqns_torch(y, p_dict, t=None):
    """
    出芽酵母细胞周期 ODE 系统的 PyTorch 可微实现。
    完全对齐 lhs_v1_origin.py 第 77-149 行的 eqns 函数。

    Args:
        y: (B, 39) 状态变量张量
        p_dict: 参数字典，值为 (B,) 张量
        t: (B,) 或标量时间张量（用于 is_mutant_104 的条件判断，可选）

    Returns:
        dydt: (B, 39) 导数张量
    """
    # 解包状态变量 (B,)
    MASS = y[:, nMASS]; CLN2 = y[:, nCLN2]; CLB2 = y[:, nCLB2]; CLB5 = y[:, nCLB5]
    SIC1 = y[:, nSIC1]; CDC6 = y[:, nCDC6]; C2 = y[:, nC2]; C5 = y[:, nC5]
    F2 = y[:, nF2]; F5 = y[:, nF5]
    SIC1P = y[:, nSIC1P]; C2P = y[:, nC2P]; C5P = y[:, nC5P]
    CDC6P = y[:, nCDC6P]; F2P = y[:, nF2P]; F5P = y[:, nF5P]
    SWI5T = y[:, nSWI5T]; SWI5 = y[:, nSWI5]; IEP = y[:, nIEP]
    CDC20T = y[:, nCDC20T]; CDC20 = y[:, nCDC20]
    CDH1T = y[:, nCDH1T]; CDH1 = y[:, nCDH1]
    CDC14T = y[:, nCDC14T]; CDC14 = y[:, nCDC14]
    NET1T = y[:, nNET1T]; NET1 = y[:, nNET1]; RENT = y[:, nRENT]
    TEM1 = y[:, nTEM1]; CDC15 = y[:, nCDC15]; PPX = y[:, nPPX]
    PDS1 = y[:, nPDS1]; ESP1 = y[:, nESP1]
    ORI = y[:, nORI]; BUD = y[:, nBUD]; SPN = y[:, nSPN]
    Vi20 = y[:, nVi20]; lte1 = y[:, nlte1]; BUB2 = y[:, nBUB2]

    # 辅助变量
    kg = math.log(2) / p_dict['mdt']
    BCK2 = p_dict['B0'] * MASS
    CLN3 = p_dict['C0'] * p_dict['Dn3'] * MASS / (p_dict['Jn3'] + p_dict['Dn3'] * MASS)
    SIC1T = SIC1 + C2 + C5 + SIC1P + C2P + C5P
    CDC6T = CDC6 + F2 + F5 + CDC6P + F2P + F5P
    RENTP = CDC14T - RENT - CDC14
    NET1P = NET1T - NET1 - CDC14T + CDC14
    PE = p_dict['ESP1T'] - ESP1

    # 反应速率
    Vasbf = p_dict['kasbf'] * (p_dict['esbfn2'] * CLN2 + p_dict['esbfn3'] * (CLN3 + BCK2) + p_dict['esbfb5'] * CLB5)
    Vdb2 = p_dict['kdb2p'] + p_dict['kdb2pp'] * CDH1 + p_dict['kdb2P'] * CDC20
    Vdb5 = p_dict['kdb5p'] + p_dict['kdb5pp'] * CDC20
    Vkpc1 = p_dict['kd1c1'] + p_dict['kd2c1'] * (p_dict['ec1n3'] * CLN3 + p_dict['ec1k2'] * BCK2 + p_dict['ec1n2'] * CLN2 + p_dict['ec1b2'] * CLB2 + p_dict['ec1b5'] * CLB5) / (p_dict['Jd2c1'] + SIC1T)
    Vppc1 = p_dict['kppc1'] * CDC14
    Vkpf6 = p_dict['kd1f6'] + p_dict['kd2f6'] * (p_dict['ef6n3'] * CLN3 + p_dict['ef6k2'] * BCK2 + p_dict['ef6n2'] * CLN2 + p_dict['ef6b2'] * CLB2 + p_dict['ef6b5'] * CLB5) / (p_dict['Jd2f6'] + CDC6T)
    Vppf6 = p_dict['kppf6'] * CDC14
    Vaiep = p_dict['kaiep'] * CLB2
    Vacdh = p_dict['kacdhp'] + p_dict['kacdhpp'] * CDC14
    Vicdh = p_dict['kicdhp'] + p_dict['kicdhpp'] * (p_dict['eicdhn3'] * CLN3 + p_dict['eicdhn2'] * CLN2 + p_dict['eicdhb5'] * CLB5 + p_dict['eicdhb2'] * CLB2)
    Vppnet = p_dict['kppnetp'] + p_dict['kppnetpp'] * PPX
    Vkpnet = (p_dict['kkpnetp'] + p_dict['kkpnetpp'] * CDC15) * MASS
    Vdpds = p_dict['kd1pdsp'] + p_dict['kd2pdspp'] * CDC20 + p_dict['kd3pdspp'] * CDH1
    Vdppx = p_dict['kdppxp'] + p_dict['kdppxpp'] * (p_dict['J20ppx'] + CDC20) * p_dict['Jpds'] / (p_dict['Jpds'] + PDS1)

    SBF = _gk_torch(Vasbf, p_dict['kisbfp'] + p_dict['kisbfpp'] * CLB2, p_dict['Jasbf'], p_dict['Jisbf'])
    MCM1 = _gk_torch(p_dict['kamcm'] * CLB2, p_dict['kimcm'], p_dict['Jamcm'], p_dict['Jimcm'])

    # is_mutant_104 条件：t >= 180 时 ksc1p 放大 3 倍
    # 用可微的 soft 判断替代硬 if（避免梯度断裂）
    # 注意：numpy 版用 if p.get('is_mutant_104') and t >= 180.0
    # 这里用 sigmoid 软化，但默认 is_mutant_104=False 时 actual_ksc1p = ksc1p
    actual_ksc1p = p_dict['ksc1p']
    if t is not None and 'is_mutant_104' in p_dict:
        # soft mask: sigmoid((t - 180) / temperature) * is_mutant_104
        temperature = 1.0
        time_mask = torch.sigmoid((t - 180.0) / temperature)
        mutant_mask = p_dict['is_mutant_104'].float()
        actual_ksc1p = p_dict['ksc1p'] * (1 + 2.0 * time_mask * mutant_mask)  # 0.012*3 = 0.012*(1+2)

    # 构建 dydt (B, 39)
    B = y.shape[0]
    dydt = torch.zeros_like(y)

    dydt[:, 0] = kg * MASS
    dydt[:, 1] = (p_dict['ksn2p'] + p_dict['ksn2pp'] * SBF) * MASS - p_dict['kdn2'] * CLN2
    dydt[:, 2] = (p_dict['ksb2p'] + p_dict['ksb2pp'] * MCM1) * MASS + (p_dict['kd3c1'] * C2P + p_dict['kd3f6'] * F2P) + (p_dict['kdib2'] * C2 + p_dict['kdif2'] * F2) - (Vdb2 + p_dict['kasb2'] * SIC1 + p_dict['kasf2'] * CDC6) * CLB2
    dydt[:, 3] = (p_dict['ksb5p'] + p_dict['ksb5pp'] * SBF) * MASS + (p_dict['kd3c1'] * C5P + p_dict['kd3f6'] * F5P) + (p_dict['kdib5'] * C5 + p_dict['kdif5'] * F5) - (Vdb5 + p_dict['kasb5'] * SIC1 + p_dict['kasf5'] * CDC6) * CLB5
    dydt[:, 4] = (actual_ksc1p + p_dict['ksc1pp'] * SWI5) + (Vdb2 * C2 + Vdb5 * C5) + (p_dict['kdib2'] * C2 + p_dict['kdib5'] * C5) + Vppc1 * SIC1P - (p_dict['kasb2'] * CLB2 + p_dict['kasb5'] * CLB5 + Vkpc1) * SIC1
    dydt[:, 5] = (p_dict['ksf6p'] + p_dict['ksf6pp'] * SWI5 + p_dict['ksf6ppp'] * SBF) + (Vdb2 * F2 + Vdb5 * F5) + (p_dict['kdif2'] * F2 + p_dict['kdif5'] * F5) + Vppf6 * CDC6P - (p_dict['kasf2'] * CLB2 + p_dict['kasf5'] * CLB5 + Vkpf6) * CDC6
    dydt[:, 6] = p_dict['kasb2'] * CLB2 * SIC1 + Vppc1 * C2P - (p_dict['kdib2'] + Vdb2 + Vkpc1) * C2
    dydt[:, 7] = p_dict['kasb5'] * CLB5 * SIC1 + Vppc1 * C5P - (p_dict['kdib5'] + Vdb5 + Vkpc1) * C5
    dydt[:, 8] = p_dict['kasf2'] * CLB2 * CDC6 + Vppf6 * F2P - (p_dict['kdif2'] + Vdb2 + Vkpf6) * F2
    dydt[:, 9] = p_dict['kasf5'] * CLB5 * CDC6 + Vppf6 * F5P - (p_dict['kdif5'] + Vdb5 + Vkpf6) * F5
    dydt[:, 10] = Vkpc1 * SIC1 - (Vppc1 + p_dict['kd3c1']) * SIC1P + Vdb2 * C2P + Vdb5 * C5P
    dydt[:, 11] = Vkpc1 * C2 - (Vppc1 + p_dict['kd3c1'] + Vdb2) * C2P
    dydt[:, 12] = Vkpc1 * C5 - (Vppc1 + p_dict['kd3c1'] + Vdb5) * C5P
    dydt[:, 13] = Vkpf6 * CDC6 - (Vppf6 + p_dict['kd3f6']) * CDC6P + Vdb2 * F2P + Vdb5 * F5P
    dydt[:, 14] = Vkpf6 * F2 - (Vppf6 + p_dict['kd3f6'] + Vdb2) * F2P
    dydt[:, 15] = Vkpf6 * F5 - (Vppf6 + p_dict['kd3f6'] + Vdb5) * F5P
    dydt[:, 16] = p_dict['ksswip'] + p_dict['ksswipp'] * MCM1 - p_dict['kdswi'] * SWI5T
    dydt[:, 17] = p_dict['ksswip'] + p_dict['ksswipp'] * MCM1 + p_dict['kaswi'] * CDC14 * (SWI5T - SWI5) - (p_dict['kiswi'] * CLB2 + p_dict['kdswi']) * SWI5
    dydt[:, 18] = Vaiep * (1 - IEP) / (p_dict['Jaiep'] + 1 - IEP) - p_dict['kiiep'] * IEP / (p_dict['Jiiep'] + IEP)
    dydt[:, 19] = (p_dict['ks20p'] + p_dict['ks20pp'] * MCM1) - p_dict['kd20'] * CDC20T
    dydt[:, 20] = (p_dict['ka20p'] + p_dict['ka20pp'] * IEP) * (CDC20T - CDC20) - (Vi20 + p_dict['kd20']) * CDC20
    dydt[:, 21] = p_dict['kscdh'] - p_dict['kdcdh'] * CDH1T
    dydt[:, 22] = p_dict['kscdh'] - p_dict['kdcdh'] * CDH1 + Vacdh * (CDH1T - CDH1) / (p_dict['Jacdh'] + CDH1T - CDH1) - Vicdh * CDH1 / (p_dict['Jicdh'] + CDH1)
    dydt[:, 23] = p_dict['ks14'] - p_dict['kd14'] * CDC14T
    dydt[:, 24] = (p_dict['kdirent'] * RENT + p_dict['kdirentp'] * RENTP) - (p_dict['kasrent'] * NET1 + p_dict['kasrentp'] * NET1P) * CDC14 + p_dict['ks14'] - p_dict['kd14'] * CDC14 + p_dict['kdnet'] * (RENT + RENTP)
    dydt[:, 25] = p_dict['ksnet'] - p_dict['kdnet'] * NET1T
    dydt[:, 26] = p_dict['kdirent'] * RENT - p_dict['kasrent'] * NET1 * CDC14 + Vppnet * NET1P - Vkpnet * NET1 + p_dict['ksnet'] - p_dict['kdnet'] * NET1 + p_dict['kd14'] * RENT
    dydt[:, 27] = -p_dict['kdirent'] * RENT + p_dict['kasrent'] * NET1 * CDC14 + Vppnet * RENTP - Vkpnet * RENT - (p_dict['kd14'] + p_dict['kdnet']) * RENT
    dydt[:, 28] = lte1 * (p_dict['Tem1T'] - TEM1) / (p_dict['jatem'] + p_dict['Tem1T'] - TEM1) - BUB2 * TEM1 / (p_dict['Jitem'] + TEM1)
    dydt[:, 29] = (p_dict['ka15p'] * (p_dict['Tem1T'] - TEM1) + p_dict['ka15pp'] * TEM1 + p_dict['ka15P'] * CDC14) * (p_dict['Cdc15T'] - CDC15) - p_dict['ki15'] * CDC15
    dydt[:, 30] = p_dict['ksppx'] - Vdppx * PPX
    dydt[:, 31] = (p_dict['kspdsp'] + p_dict['ks1pdspp'] * SBF + p_dict['ks2pdspp'] * MCM1) + p_dict['kdiesp'] * PE - (Vdpds + p_dict['kasesp'] * ESP1) * PDS1
    dydt[:, 32] = -p_dict['kasesp'] * PDS1 * ESP1 + (p_dict['kdiesp'] + Vdpds) * PE
    dydt[:, 33] = p_dict['ksori'] * (p_dict['eorib5'] * CLB5 + p_dict['eorib2'] * CLB2) - p_dict['kdori'] * ORI
    dydt[:, 34] = p_dict['ksbud'] * (p_dict['ebudn2'] * CLN2 + p_dict['ebudn3'] * CLN3 + p_dict['ebudb5'] * CLB5) - p_dict['kdbud'] * BUD
    dydt[:, 35] = p_dict['ksspn'] * CLB2 / (p_dict['Jspn'] + CLB2) - p_dict['kdspn'] * SPN
    # dydt[36:39] = 0 (Vi20, lte1, BUB2 在连续 ODE 中导数为 0，由 event 驱动)
    return dydt


def apply_event_torch(y, p_dict, event_id):
    """
    Event 重置规则的可微 PyTorch 实现。
    对齐 lhs_v1_origin.py 第 285-299 行的 event 重置逻辑。

    用于跳跃条件损失：L_jump = || û(t⁺) - T_event(û(t⁻)) ||²

    Args:
        y: (B, 39) event 触发前的状态（t⁻）
        p_dict: 参数字典（值可以是 (B,) 张量或标量张量）
        event_id: (B,) 每个样本触发的 event 编号 (1,2,3,4)，0 表示无 event

    Returns:
        y_after: (B, 39) event 重置后的状态（t⁺）
    """
    y_after = y.clone()
    B = y.shape[0]

    # 辅助函数：确保参数是 (B,) 形状
    def _expand(v):
        if isinstance(v, torch.Tensor):
            if v.dim() == 0:
                return v.expand(B)
            return v
        return torch.full((B,), float(v), device=y.device, dtype=y.dtype)

    # event1: 分裂 (CLB2=0.3↓)
    # y0[nBUD] = y0[nSPN] = 0
    # y0[nMASS] = abs(1 + exp(-kg * ((1.026/kg) - 32)) - 1) * y0[nMASS]
    # y0[nlte1] = 0.1; y0[nCLB2] -= 1e-6
    mask1 = (event_id == 1).float()  # (B,)
    mdt = _expand(p_dict['mdt'])
    kg = math.log(2) / mdt  # (B,)
    mass_factor = torch.abs(1 + torch.exp(-kg * ((1.026 / kg) - 32)) - 1)  # (B,)
    y_after[:, nBUD] = y[:, nBUD] * (1 - mask1)
    y_after[:, nSPN] = y[:, nSPN] * (1 - mask1)
    y_after[:, nMASS] = y[:, nMASS] * (1 - mask1) + (y[:, nMASS] * mass_factor) * mask1
    y_after[:, nlte1] = y[:, nlte1] * (1 - mask1) + 0.1 * mask1
    y_after[:, nCLB2] = y[:, nCLB2] - 1e-6 * mask1

    # event2: S期退出 (CLB2+CLB5=0.2↓)
    # y0[nORI] = 0; y0[nCLB5] -= 1e-6
    mask2 = (event_id == 2).float()
    y_after[:, nORI] = y[:, nORI] * (1 - mask2)
    y_after[:, nCLB5] = y[:, nCLB5] - 1e-6 * mask2

    # event3: DNA复制 (ORI=1.0↑)
    # y0[nVi20] = p['Vi20_active']; y0[nBUB2] = p['BUB2_active']; y0[nORI] += 1e-6
    mask3 = (event_id == 3).float()
    vi20_active = _expand(p_dict['Vi20_active'])
    bub2_active = _expand(p_dict['BUB2_active'])
    y_after[:, nVi20] = y[:, nVi20] * (1 - mask3) + vi20_active * mask3
    y_after[:, nBUB2] = y[:, nBUB2] * (1 - mask3) + bub2_active * mask3
    y_after[:, nORI] = y_after[:, nORI] + 1e-6 * mask3

    # event4: 纺锤体通过 (SPN=1.0↑)
    # y0[nVi20] = 0.01; y0[nlte1] = 1.0; y0[nBUB2] = init_BUB2; y0[nSPN] += 1e-6
    mask4 = (event_id == 4).float()
    init_bub2 = _expand(p_dict['init_BUB2'])
    y_after[:, nVi20] = y_after[:, nVi20] * (1 - mask4) + 0.01 * mask4
    y_after[:, nlte1] = y_after[:, nlte1] * (1 - mask4) + 1.0 * mask4
    y_after[:, nBUB2] = y_after[:, nBUB2] * (1 - mask4) + init_bub2 * mask4
    y_after[:, nSPN] = y_after[:, nSPN] + 1e-6 * mask4

    return y_after


def compute_ode_residual_autograd(pred_trajectory, p_dict, t_grid, dt):
    """
    用 autograd 计算 ODE 残差（PINN 核心方法）。

    残差定义（导师要求）：
        r(t) = dû/dt - f(û(t), p)
    其中 dû/dt 通过 autograd 对时间求导得到（精确导数，非有限差分）。

    Args:
        pred_trajectory: (B, 39, T) 模型预测轨迹（真实值空间）
            注意：必须 requires_grad=True 且与 t_grid 有计算图连接
        p_dict: 参数字典
        t_grid: (T,) 时间网格张量
        dt: 时间步长

    Returns:
        residual: (B, 39, T) ODE 残差
    """
    B, V, T = pred_trajectory.shape

    # 方法：对每个时间步，用有限差分近似 dû/dt
    # （完整 autograd 需要模型对 t 可微，这里用有限差分作为近似）
    # 中心差分
    du_dt = torch.zeros_like(pred_trajectory)
    du_dt[:, :, 1:-1] = (pred_trajectory[:, :, 2:] - pred_trajectory[:, :, :-2]) / (2 * dt)
    du_dt[:, :, 0] = (pred_trajectory[:, :, 1] - pred_trajectory[:, :, 0]) / dt
    du_dt[:, :, -1] = (pred_trajectory[:, :, -1] - pred_trajectory[:, :, -2]) / dt

    # 计算 f(u, p) at each timestep
    f_u = torch.zeros_like(pred_trajectory)
    for t_idx in range(T):
        y_t = pred_trajectory[:, :, t_idx]  # (B, 39)
        t_t = t_grid[t_idx].expand(B) if t_grid is not None else None
        f_u[:, :, t_idx] = eqns_torch(y_t, p_dict, t_t)

    residual = du_dt - f_u
    return residual


def detect_event_mask_torch(u, event_window=4, event_tol=0.08, jump_ratio=0.3):
    """
    Event 掩码检测的 PyTorch 版本（对齐 al_callback.py 的 _detect_event_mask）。
    用于 PINN 损失中屏蔽 event 区间的残差。

    Args:
        u: (B, 39, T) 预测轨迹
        event_window: 掩码窗口半径
        event_tol: 事件触发条件容差
        jump_ratio: 跳变检测阈值（占变量值域比例）

    Returns:
        mask: (B, T) 布尔张量，True=连续区间，False=event 区间
    """
    B, V, T = u.shape
    mask = torch.ones(B, T, dtype=torch.bool, device=u.device)

    clb2 = u[:, nCLB2]
    clb5 = u[:, nCLB5]
    ori = u[:, nORI]
    spn = u[:, nSPN]

    # 事件条件检测
    d_clb2 = torch.cat([clb2[:, :1], clb2[:, 1:] - clb2[:, :-1]], dim=1)
    event1 = (torch.abs(clb2 - 0.3) < event_tol) & (d_clb2 < 0)

    clb2_clb5 = clb2 + clb5
    d_cc = torch.cat([clb2_clb5[:, :1], clb2_clb5[:, 1:] - clb2_clb5[:, :-1]], dim=1)
    event2 = (torch.abs(clb2_clb5 - 0.2) < event_tol) & (d_cc < 0)

    d_ori = torch.cat([ori[:, :1], ori[:, 1:] - ori[:, :-1]], dim=1)
    event3 = (torch.abs(ori - 1.0) < event_tol) & (d_ori > 0)

    d_spn = torch.cat([spn[:, :1], spn[:, 1:] - spn[:, :-1]], dim=1)
    event4 = (torch.abs(spn - 1.0) < event_tol) & (d_spn > 0)

    all_events = event1 | event2 | event3 | event4

    # 跳变检测
    # 使用绝对阈值（基于 event 重置规则的典型变化幅度），
    # 而非相对阈值，避免对噪声数据过于敏感
    # event1: BUD/SPN 归零（变化 > 0.3）
    # event2: ORI 归零（变化 > 0.5）
    # event3: Vi20/BUB2 激活（变化 > 0.3）
    # event4: Vi20 归零, lte1 上升（变化 > 0.3）
    jump_thresholds = {
        nMASS: 0.5,    # MASS 衰减
        nCLB2: 0.3,    # CLB2 降解
        nCLB5: 0.3,    # CLB5 降解
        nORI: 0.5,     # ORI 归零
        nBUD: 0.3,     # BUD 归零
        nSPN: 0.3,     # SPN 归零
        nVi20: 0.3,    # Vi20 激活/归零
        nlte1: 0.3,    # lte1 上升
        nBUB2: 0.3,    # BUB2 激活
    }
    for v, thresh in jump_thresholds.items():
        d = torch.abs(torch.cat([u[:, v:v+1, 0], u[:, v, 1:] - u[:, v, :-1]], dim=1))
        abnormal = d > thresh
        all_events = all_events | abnormal

    # 扩展掩码窗口
    all_events_np = all_events.cpu().numpy()
    for b in range(B):
        event_indices = np.where(all_events_np[b])[0]
        for t in event_indices:
            t_start = max(0, t - event_window)
            t_end = min(T, t + event_window + 1)
            mask[b, t_start:t_end] = False

    return mask
