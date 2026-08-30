import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from covid19_model_reis_2021 import modelo, pars, params_ajs, carrega_dados, monta_y0

nk, viremia, igm, igg, il6 = carrega_dados()

nk      = (nk['x'].values, nk['y'].values)
viremia = (viremia[viremia.type == 'mean']['x'].values, viremia[viremia.type == 'mean']['y'].values)
igm     = (igm[igm.type == 'mean']['x'].values, igm[igm.type == 'mean']['y'].values)
igg     = (igg[igg.type == 'mean']['x'].values, igg[igg.type == 'mean']['y'].values)
il6     = (il6[il6.type == 'mean']['x'].values, il6[il6.type == 'mean']['y'].values)

nk['y'] = nk['y'] / 1000
# Transformação Logarítimica
e = 1.0

nk_log = np.log10(nk[1] + e)
viremia_log = np.log10(viremia[1] + e)
igm_log = np.log10(igm[1] + e)
igg_log = np.log10(igg[1] + e)
il6_log = np.log10(il6[1] + e)

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

# Z-score
z_nk = (nk_log - nk_media)/nk_dv
z_viremia = (viremia_log - viremia_media)/viremia_dv
z_igm = (igm_log - igm_media)/igm_dv
z_igg = (igg_log - igg_media)/igg_dv
z_il6 = (il6_log - il6_media)/il6_dv


# Espaço de busca params_ajs
bounds = [
    [1.2, 1.6],         # pi_v
    [0.01, 0.03],       # kv1
    [1.4e-5, 1.6e-5],   # kv2
    [0.01, 1e0],        # beta_ap
    [0.0005, 0.065],    # gamma_apm
    [1.4e-6, 1.5e-4],   # beta_th
    [5e-9, 6e-7],       # pi_th
    [0.27, 27e0],       # delta_th
    [0.75, 70e0],       # alpha_tk
    [0.00001, 0.0015],  # beta_tk
    [9.6e-9, 9.6e-7],   # pi_tk
    [0.0002, 0.04],     # delta_tk
    [0.00008, 0.00009], # pi_pl
    [0.004, 0.55],      # qn
    [0.06, 7e0],        # dn
    [0.0005, 0.06],     # gamma_iNK
    [0.0008, 0.09],     # pi_cNK
    [1e0, 1e6],         # V0
    [1.3e4, 1.5e6],     # NK0
    [1.5e5, 3.5e6]      # Nmax
]
"""
bounds = []

for i in params_ajs:
    val = pars[i]
    bounds.append((val*0.1, val*10))"""

# Tempo de simulação
t0 = 0.0
tf = 37.0
t = np.arange(t0, tf, 0.1)
t_span = (t0, tf)


# Função Objetivo
def model_objetivo(params):
    p = pars.copy()
    for i, key in enumerate(params_ajs):
        p[key] = params[i]
        
    y0_atualizado = monta_y0(p)
    sol = solve_ivp(modelo, t_span, y0_atualizado, args = (p,), method='Radau', t_eval = t)

    if not sol.success or np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
        return 1e18  # Retorna um erro muito alto para o otimizador descartar esses parâmetros
    # Interpolando os passos de tempo
    nk_interp = np.interp(nk[0], t, sol.y[15])
    viremia_interp = np.interp(viremia[0], t, sol.y[0])
    igm_interp = np.interp(igm[0], t, sol.y[12])
    igg_interp = np.interp(igg[0], t, sol.y[13])
    il6_interp = np.interp(il6[0], t, sol.y[14])

    # Evitando valores negativos no modelo
    nk_interp = np.clip(nk_interp, 1e-12, None)
    viremia_interp = np.clip(viremia_interp, 1e-12, None)
    igm_interp = np.clip(igm_interp, 1e-12, None)
    igg_interp = np.clip(igg_interp, 1e-12, None)
    il6_interp = np.clip(il6_interp, 1e-12, None)

    nk_interp = np.log10(nk_interp + e)
    viremia_interp = np.log10(viremia_interp + e)
    igm_interp = np.log10(igm_interp + e)
    igg_interp = np.log10(igg_interp + e)
    il6_interp = np.log10(il6_interp + e)
    
    # Z-score
    z_m_nk = (nk_interp - nk_media) / nk_dv
    z_m_viremia = (viremia_interp - viremia_media) / viremia_dv
    z_m_igm = (igm_interp - igm_media) / igm_dv
    z_m_igg = (igg_interp - igg_media) / igg_dv
    z_m_il6 = (il6_interp - il6_media) / il6_dv

    # Erro total
    res_nk = z_m_nk - z_nk
    res_viremia = z_m_viremia - z_viremia
    res_igm = z_m_igm - z_igm
    res_igg = z_m_igg - z_igg
    res_il6 = z_m_il6 - z_il6

    return float((np.mean(res_nk**2)) + (np.mean(res_viremia**2)) + (np.mean(res_igm**2)) + 
                 (np.mean(res_igg**2)) + (np.mean(res_il6**2)))



def model_adj(params): 
    return model_objetivo(params)

if __name__ == '__main__':
    # Evolução Diferencial
    result = differential_evolution(
        model_adj, 
        bounds, 
        strategy = 'best1bin', 
        popsize = 20, 
        mutation = (0.5, 1),  
        recombination = 0.7, 
        disp = True,
        workers = -1) # Usar todos os núcleos de processamento

    # Salvando os resultados em um .npy
    np.save(r'C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\params_otimos\parametros_otimos.npy', result.x)











