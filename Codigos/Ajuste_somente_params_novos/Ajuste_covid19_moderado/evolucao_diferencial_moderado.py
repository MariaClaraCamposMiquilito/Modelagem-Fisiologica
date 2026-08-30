import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from covid19_modelo_moderado import modelo, y0, pars, params_ajs, carrega_dados
import warnings

warnings.filterwarnings('ignore', category = RuntimeWarning)

nk, viremia, igm, igg, il6, tcd4, tcd8, cellB = carrega_dados()

nk      = (nk[nk.type == 'mean']['x'].values, nk[nk.type == 'mean']['y'].values)
viremia = (viremia[viremia.type == 'mean']['x'].values, viremia[viremia.type == 'mean']['y'].values)
igm     = (igm[igm.type == 'mean']['x'].values, igm[igm.type == 'mean']['y'].values)
igg     = (igg[igg.type == 'mean']['x'].values, igg[igg.type == 'mean']['y'].values)
il6     = (il6[il6.type == 'mean']['x'].values, il6[il6.type == 'mean']['y'].values)
tcd4    = (tcd4[tcd4.type == 'mean']['x'].values, tcd4[tcd4.type == 'mean']['y'].values)
tcd8    = (tcd8[tcd8.type == 'mean']['x'].values, tcd8[tcd8.type == 'mean']['y'].values)
cellB   = (cellB[cellB.type == 'mean']['x'].values, cellB[cellB.type == 'mean']['y'].values)

# Transformação Logarítimica
e = 1.0

nk_log = np.log10(nk[1] + e)
viremia_log = np.log10(viremia[1] + e)
igm_log = np.log10(igm[1] + e)
igg_log = np.log10(igg[1] + e)
il6_log = np.log10(il6[1] + e)
tcd4_log = np.log10(tcd4[1] + e)
tcd8_log = np.log10(tcd8[1] + e)
cellB_log = np.log10(cellB[1] + e)

# Normalização - média e desvio-padrão
nk_media = nk_log.mean()
nk_dv = nk_log.std(ddof=1)

viremia_media = viremia_log.mean()
viremia_dv = viremia_log.std(ddof = 1)

igm_media = igm_log.mean()
igm_dv = igm_log.std(ddof = 1)

igg_media = igg_log.mean()
igg_dv = igg_log.std(ddof = 1)

il6_media = il6_log.mean()
il6_dv = il6_log.std(ddof = 1)

tcd4_media = tcd4_log.mean()
tcd4_dv = tcd4_log.std(ddof = 1)

tcd8_media = tcd8_log.mean()
tcd8_dv = tcd8_log.std(ddof = 1)

cellB_media = cellB_log.mean()
cellB_dv = cellB_log.std(ddof = 1)

# Z-score
z_nk = (nk_log - nk_media)/nk_dv
z_viremia = (viremia_log - viremia_media)/viremia_dv
z_igm = (igm_log - igm_media)/igm_dv
z_igg = (igg_log - igg_media)/igg_dv
z_il6 = (il6_log - il6_media)/il6_dv
z_tcd4 = (tcd4_log - tcd4_media)/tcd4_dv
z_tcd8 = (tcd8_log - tcd8_media)/tcd8_dv
z_cellB = (cellB_log - cellB_media)/cellB_dv

bounds = [
    (0.8, 1.7),          # 0 pi_v
    (0.001, 0.10),       # 1 kv3
    (0.001, 0.5),        # 2 beta_ap
    (0.001, 5.0),        # 3 cap1
    (1e6, 6e7),          # 4 cap2
    (0.1, 0.9),          # 5 beta_apm
    (0.1, 0.9),          # 6 beta_th
    (0.0005, 0.001),     # 7 delta_th
    (3e-5, 2.5e-4),      # 8 beta_tk
    (0.005, 0.01),       # 9 gamma_c
    (0.50, 0.95),        # 10 dn
    (0.05, 0.5),         # 11 pi_cNK
    (1e6, 5e6),          # 12 Ap0
    (3.5e5, 7e5),        # 13 TkN0
    (1.2e5, 4e5),        # 14 B0
]
"""

bounds = [
    (0.65, 1.20),        # 0 pi_v
    (0.12, 0.30),        # 1 kv3
    (0.015, 0.060),      # 2 beta_ap
    (2.5, 6.0),          # 3 cap1
    (7e6, 1.3e7),        # 4 cap2

    (0.006, 0.018),      # 5 beta_apm

    (5e-5, 8e-4),        # 6 beta_th
    (0.015, 0.080),      # 7 delta_th

    (3e-5, 2.5e-4),      # 8 beta_tk

    (80.0, 500.0),       # 9 gamma_c
    (0.50, 0.95),        # 10 dn
    (0.003, 0.012),      # 11 pi_cNK

    (7e5, 1.2e6),        # 12 Ap0
    (3.5e5, 7e5),        # 13 TkN0
    (1.2e5, 4e5),        # 14 B0
]
bounds = []

for i in params_ajs:
    bounds.append((pars[i] * 0.1, pars[i] * 1.5))

"""

# Tempo de simulação
t0 = 0.0
tf = 70.0
t = np.arange(t0, tf, 0.1)
t_span = (t0, tf)

# Função Objetivo
def model_objetivo(params):
    p = pars.copy()
    for i, key in enumerate(params_ajs):
        p[key] = params[i]
        
    try:

        sol = solve_ivp(modelo, t_span, y0, args=(p,), method='Radau', t_eval=t)
        if not sol.success or np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
            return 1e18
            
    except Exception:
        return 1e18
    
    # Interpolando os passos de tempo
    nk_interp = np.interp(nk[0] + 4.15, t, sol.y[15])
    viremia_interp = np.interp(viremia[0] + 4.15, t, sol.y[0])
    igm_interp = np.interp(igm[0] + 4.15, t, sol.y[12])
    igg_interp = np.interp(igg[0] + 4.15, t, sol.y[13])
    il6_interp = np.interp(il6[0] + 4.15, t, sol.y[14])
    tcd4_interp = np.interp(tcd4[0] + 4.15, t, sol.y[5])
    tcd8_interp = np.interp(tcd8[0] + 4.15, t, sol.y[7])
    cellB_interp = np.interp(cellB[0] + 4.15, t, (sol.y[8]) + sol.y[11])

    # Evitando valores negativos no modelo
    nk_interp = np.clip(nk_interp, 1e-12, None)
    viremia_interp = np.clip(viremia_interp, 1e-12, None)
    igm_interp = np.clip(igm_interp, 1e-12, None)
    igg_interp = np.clip(igg_interp, 1e-12, None)
    il6_interp = np.clip(il6_interp, 1e-12, None)
    tcd4_interp = np.clip(tcd4_interp, 1e-12, None)
    tcd8_interp = np.clip(tcd8_interp, 1e-12, None)
    cellB_interp = np.clip(cellB_interp, 1e-12, None)

    nk_interp = np.log10(nk_interp + e)
    viremia_interp = np.log10(viremia_interp + e)
    igm_interp = np.log10(igm_interp + e)
    igg_interp = np.log10(igg_interp + e)
    il6_interp = np.log10(il6_interp + e)
    tcd4_interp  = np.log10(tcd4_interp * 1e3 + e)
    tcd8_interp  = np.log10(tcd8_interp * 1e3 + e)
    cellB_interp = np.log10(cellB_interp * 1e3 + e)
    
    # Z-score
    z_m_nk = (nk_interp - nk_media) / nk_dv
    z_m_viremia = (viremia_interp - viremia_media) / viremia_dv
    z_m_igm = (igm_interp - igm_media) / igm_dv
    z_m_igg = (igg_interp - igg_media) / igg_dv
    z_m_il6 = (il6_interp - il6_media) / il6_dv
    z_m_tcd4 = (tcd4_interp - tcd4_media) / tcd4_dv
    z_m_tcd8 = (tcd8_interp - tcd8_media) / tcd8_dv
    z_m_cellB = (cellB_interp - cellB_media) / cellB_dv


    # Erro total 
    res_nk = z_m_nk - z_nk
    res_viremia = z_m_viremia - z_viremia
    res_igm = z_m_igm - z_igm
    res_igg = z_m_igg - z_igg
    res_il6 = z_m_il6 - z_il6
    res_tcd4 = z_m_tcd4 - z_tcd4
    res_tcd8 = z_m_tcd8 - z_tcd8
    res_cellB = z_m_cellB - z_cellB


    #return float(np.mean(res_viremia**2) + np.mean(res_nk**2) + np.mean(res_igm**2) + np.mean(res_igg**2) + 
     #            np.mean(res_il6**2) + np.mean(res_tcd4**2) + np.mean(res_tcd8**2) + np.mean(res_cellB**2))
    return float(np.mean(res_tcd4**2))


def model_adj(params): 
    return model_objetivo(params)

if __name__ == '__main__':
    
    # Evolução Diferencial
    result = differential_evolution(
        model_adj, 
        bounds, 
        strategy = 'best1bin',
        popsize = 70,
        mutation = (0.5, 1),
        recombination = 0.7,
        disp = True,
        workers = 10) # Usar todos os núcleos de processamento

    # Salvando os resultados em um .npy
    np.save(r'C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\params_otimos\parametros_otimos.npy', result.x)




 





