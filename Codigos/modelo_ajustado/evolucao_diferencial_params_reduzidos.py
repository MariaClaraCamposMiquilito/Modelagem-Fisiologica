import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from modelo_covid19_ajustado import modelo, carrega_dados, pars, y0, params_ajs

# Arrumando os dados experimentais -> pega o x e o y dos datasets referente às médias
viremia, il6, igg, igm, nk = carrega_dados()
data_v = (viremia[viremia.type == 'mean']['x'].values, viremia[viremia.type == 'mean']['y'].values)
data_c = (il6[il6.type == 'mean']['x'].values, il6[il6.type == 'mean']['y'].values)
data_g = (igg[igg.type == 'mean']['x'].values, igg[igg.type == 'mean']['y'].values)
data_m = (igm[igm.type == 'mean']['x'].values, igm[igm.type == 'mean']['y'].values)
data_n = (nk['x'].values, nk['y'].values)

keys_pars = [k for k in params_ajs]

# Espaço de busca

bounds = [
    [1.0, 1e3],          # V0
    [0.8, 2],            # pi_v
    [1e-5, 1e-3],        # kv1
    [1e-7, 1e-5],        # kv2
    [1e-4, 0.05],        # kv3
    [1e-5, 1.0],         # beta_ap
    [1e-3, 1.0],         # beta_apm
    [1e-6, 1e-4],        # beta_tke
    [4e-4, 5e-2],        # gama_apm
    [1e-6, 1e-4],        # pi_bm1
    [9e-3, 9e-1],        # pi_ps
    [1.0, 500.0],        # pi_capm
    [5e-3, 1.0],         # pi_ci
    [1e-5, 1.0],         # pi_ctke
    [10.0, 1000.0],      # gama_c
    [0.05, 0.5e1],       # qn
    [0.007, 0.7],        # dn
    [6e-7, 6e-4],        # gamma_ink
    #[1e-4, 1e4],        # gamma_itk
    [0.0001, 1.5],       # pi_cnk
    [1e4, 1e6],          # NK0
    [3.0e5, 3.0e7]       # Nmax
]


# Tempo de simulação
tf = 37.0 # dias
dt = 0.01
N = int(tf / dt)
t = np.linspace(0, tf, N)

# Estatística dos dados
eps = 1.0

## Transformando os dados em Log10
data_v_log = np.log10(data_v[1] + eps)
data_c_log = np.log10(data_c[1] + eps)
data_g_log = np.log10(data_g[1] + eps)
data_m_log = np.log10(data_m[1] + eps)
data_n_log = np.log10(data_n[1] + eps)

## Calculando Média
media_v = data_v_log.mean()
media_c = data_c_log.mean()
media_g = data_g_log.mean()
media_m = data_m_log.mean()
media_n = data_n_log.mean()

## Calculando desvio-padrão
dv_v = data_v_log.std(ddof = 1)
dv_c = data_c_log.std(ddof = 1)
dv_g = data_g_log.std(ddof = 1)
dv_m = data_m_log.std(ddof = 1)
dv_n = data_n_log.std(ddof = 1)

## Z_score dos dados
z_data_v = (data_v_log - media_v) / dv_v
z_data_c = (data_c_log - media_c) / dv_c
z_data_g = (data_g_log - media_g) / dv_g
z_data_m = (data_m_log - media_m) / dv_m
z_data_n = (data_n_log - media_n) / dv_n

# FUNÇÃO OBJETIVO
def modelo_objetivo(params):
    p = pars.copy()
    for i, key in enumerate(keys_pars):
        p[key] = params[i]

    y0_simulacao = [
        p['V0'], p['Ap0'], p['ApM0'], p['I0'], p['ThN0'], p['ThE0'], 
        p['TkN0'], p['TkE0'], p['B0'], p['Ps0'], p['Pl0'], p['Bm0'], 
        p['IgM0'], p['IgG0'], p['C0'], p['NK0']
    ]

    sol = solve_ivp(
        modelo, 
        [0, tf], 
        y0_simulacao, 
        args = (p,), 
        method ='Radau', # Resolver EDOs rígidas
        t_eval = t)
    
    if not sol.success or sol.y.shape[1] < len(t) or np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
        return 1e18

    # .shape[1] representa os pontos no tempo onde a solução foi calculada
    # sol.y.shape[1] < len(t) -> verifica se o solver conseguiu chegar até o final do tempo no t_eval.
    # np.any(np.isnan(sol.y)) -> verifica se tem o erro NaN
    # np.any(np.isinf(sol.y)) -> verifica se o modelo explodiu -> os valores deram infinito

    # Interpolando os passos de tempo
    # sol.y[índice] extrai a linha da variável no tempo simulado 't'
    # data_x[0] são os dias onde temos dados experimentais
    v_interp = np.interp(data_v[0], t, sol.y[0])
    m_interp = np.interp(data_m[0], t, sol.y[12])
    g_interp = np.interp(data_g[0], t, sol.y[13])
    c_interp = np.interp(data_c[0], t, sol.y[14])
    n_interp = np.interp(data_n[0], t, sol.y[15])

    # Evitando valores negativos no modelo
    v_model = np.clip(v_interp, 1e-12, None)
    m_model = np.clip(m_interp, 1e-12, None)
    g_model = np.clip(g_interp, 1e-12, None)
    c_model = np.clip(c_interp, 1e-12, None)
    n_model = np.clip(n_interp, 1e-12, None)

    # Aplicando Log no modelo
    v_model_log = np.log10(v_model + eps)
    m_model_log = np.log10(m_model + eps)
    g_model_log = np.log10(g_model + eps)
    c_model_log = np.log10(c_model + eps)
    n_model_log = np.log10(n_model + eps)

    # Z-score do modelo
    z_v_model = (v_model_log - media_v) / dv_v
    z_m_model = (m_model_log - media_m) / dv_m
    z_g_model = (g_model_log - media_g) / dv_g
    z_c_model = (c_model_log - media_c) / dv_c
    z_n_model = (n_model_log - media_n) / dv_n

    # Resíduos
    res_total = (np.mean((z_v_model - z_data_v)**2) + 
                 np.mean((z_m_model - z_data_m)**2) + 
                 np.mean((z_g_model - z_data_g)**2) + 
                 np.mean((z_c_model - z_data_c)**2) +
                 np.mean((z_n_model - z_data_n)**2))
    return float(res_total)

def model_adj(params): 
    return modelo_objetivo(params)

result = differential_evolution(
    model_adj, 
    bounds, 
    strategy = 'rand1bin', 
    popsize = 15, 
    mutation = (0.5, 1.5),  
    recombination = 0.7, 
    disp = True)


# Salvando os melhores parâmetros em .npy
np.save(r'C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\modelo_ajustado\parametros_otimos.npy', result.x)
