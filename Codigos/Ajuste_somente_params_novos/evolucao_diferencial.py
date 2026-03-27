import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from covid19_model_reis_2021 import modelo, y0, pars, params_ajs

# Extraindo os dados experimentais
def carrega_dados():
    data_nk = pd.read_csv(r'C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\meus_dados\NK_covid_severo.csv', sep = ',')
    data_viremia = pd.read_csv(r'C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\covid19_model\data\Viral_load.csv', sep = ',')
    data_igm = pd.read_csv(r'C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\covid19_model\data\IgM_data.csv', sep = ',')
    data_igg = pd.read_csv(r'C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\covid19_model\data\IgG_data.csv', sep = ',')
    data_il6 = pd.read_csv(r'C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\covid19_model\data\dataset_il6_survivor.csv', sep = ',')

    return data_nk, data_viremia, data_igm, data_igg, data_il6

nk, viremia, igm, igg, il6 = carrega_dados()

nk = (nk['x'].values, nk['y'].values)
viremia = (viremia[viremia.type == 'mean']['x'].values, viremia[viremia.type == 'mean']['y'].values)
igm = (igm[igm.type == 'mean']['x'].values, igm[igm.type == 'mean']['y'].values)
igg = (igg[igg.type == 'mean']['x'].values, igg[igg.type == 'mean']['y'].values)
il6 = (il6[il6.type == 'mean']['x'].values, il6[il6.type == 'mean']['y'].values)

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
bounds = []
for p in params_ajs:
    val = pars[p]
    bounds.append((val * 0.8, val * 1.2))

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
        
    sol = solve_ivp(modelo, t_span, y0, args = (p,), method='Radau', t_eval = t)


    if np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
        return 1e10

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

# Evolução Diferencial
result = differential_evolution(
    model_adj, 
    bounds, 
    strategy = 'rand1bin', 
    popsize = 10, 
    mutation = (0.5, 1.5),  
    recombination = 0.7, 
    disp = True)

# Salvando os resultados em um .npy
np.save(r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\params_otimos\parametros_otimos.npy")











