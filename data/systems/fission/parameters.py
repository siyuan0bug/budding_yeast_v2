"""Fission 58 exported parameters; event switch kS is excluded."""
import numpy as np
def WT_parameters():
    p = dict(
        # --- Cdc13 合成与生长 ---
        k1=0.02,            # Cdc13 一阶合成速率
        mu=0.00462,         # 指数生长率 (质量倍增 150 min)
        # --- Cdc13 降解 (Ste9 介导) ---
        V2_=0.02, V2=1.0,   # 主动降解 V2 = V2_ + V2*Ste9
        V2c_=0.02, V2c=0.5, # CR/CRP 三聚体降解 V2c = V2c_ + V2c*Ste9
        # --- Inh (PP 抑制子) 合成 ---
        k3=0.1,
        # --- Inh/PI 被 Slp1a 降解 ---
        V4_=0.01, V4=1.0,
        # --- Rum1 合成与降解 ---
        k5=0.1, k6_=0.1, k6=5.0,
        # --- Rum1/Rum1P 磷酸化 (MPFa 与 SK*mass) ---
        kp=100.0, kpp_=1.0, kpp=100.0,
        # --- 三聚体 CR/CRP 结合与解离 ---
        kJ=400.0, kJR=1.0,
        # --- Ste9 修饰 ---
        kste9=5.0, kste9r_=0.03, kste9r=8.0,
        # --- Mik1 修饰 ---
        kM=1.0, kMR_=0.01, kMR=5.0,
        # --- Wee1 修饰 ---
        kW=2.0, kWR_=0.4, kWR=1.0,
        # --- Cdc25 修饰 ---
        k25=1.0, k25R_=0.4, k25R=2.0,
        # --- Slp1 (APC) ---
        kas=0.1, kad=0.1, kaa_=0.01, kaa=0.1, kai=0.1,
        # --- Inh 磷酸化 / PI 去磷酸化 ---
        ki=50.0, kiR=0.5,
        # --- S 期开关参数 ---
        kSmax=5.0, kSmin=0.1,
        # --- 酪氨酸修饰酶最大活性 ---
        Vwee=10.0, Vwee_=0.08, Vmik=2.0, Vmik_=0.04,
        V25=10.0, V25_=0.05, Vpyp=0.07,
        # --- Michaelis 常数 (超灵敏开关) ---
        Jste9=0.01, Jste9R=0.01, Jmik=0.15, JmikR=0.15,
        Jwee=0.2, JweeR=0.2, J25=0.05, J25R=0.05,
        # --- 起始激酶 SK / MPFa eps / epsP / 复制速率 K / Pyp3 总量 ---
        SK=0.018, eps=0.05, epsP=0.025, K=0.06, Pyp3=1.0,
        # --- 离散开关状态 kS (动态, 随事件切换; 不参与参数向量导出) ---
        kS=0.1,
    )
    return p

EXPORT_KEYS = sorted([k for k in WT_parameters() if k != 'kS'])

def param_vector(p):
    return np.array([p[k] for k in EXPORT_KEYS], dtype=float)

get_default_params = WT_parameters
PARAM_KEYS = EXPORT_KEYS
