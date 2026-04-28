import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from modelo_covid import modelo, carrega_dados, pars, y0

# Arrumando os dados experimentais -> pega o x e o y dos datasets referente às médias
viremia, il6, igg, igm, nk = carrega_dados()
data_v = (viremia[viremia.type == 'mean']['x'].values, viremia[viremia.type == 'mean']['y'].values)
data_c = (il6[il6.type == 'mean']['x'].values, il6[il6.type == 'mean']['y'].values)
data_g = (igg[igg.type == 'mean']['x'].values, igg[igg.type == 'mean']['y'].values)
data_m = (igm[igm.type == 'mean']['x'].values, igm[igm.type == 'mean']['y'].values)
data_n = (nk['x'].values, nk['y'].values)

# Retirando as condições iniciais fixas do dicionário, para elas não serem ajustadas
keys_pars = [k for k in pars.keys() if k not in ['Ap0', 'ThN0', 'TkN0', 'B0', 'NK0']]

# Espaço de busca
bounds = []
for k in keys_pars:
    val = pars[k]
    if val == 0:
        bounds.append((0, 0.1))
    else:
        bounds.append(val * 0.1, val * 1.5 )

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

    sol = solve_ivp(
        modelo, 
        [0, tf], 
        y0, 
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
    popsize = 10, 
    mutation = (0.5, 1.5),  
    recombination = 0.7, 
    disp = True)


# Salvando os melhores parâmetros em .npy
path = r'C:\Users\karla\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\modelo_ajustado'
np.save(path +r'\parametros_otimos.npy', result.x)