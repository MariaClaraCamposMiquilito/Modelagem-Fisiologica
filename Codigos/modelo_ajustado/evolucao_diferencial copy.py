import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from modelo_covid19_ajustado import modelo, carrega_dados, pars, y0

# Arrumando os dados experimentais -> pega o x e o y dos datasets referente às médias
viremia, il6, igg, igm, nk = carrega_dados()
# Aqui já aplica log na viremia, e nos anticorpos
data_v = (viremia[viremia.type == 'mean']['x'].values, np.log10(viremia[viremia.type == 'mean']['y'].values + 1))
data_c = (il6[il6.type == 'mean']['x'].values, il6[il6.type == 'mean']['y'].values)
data_g = (igg[igg.type == 'mean']['x'].values, np.log2(igg[igg.type == 'mean']['y'].values + 1))
data_m = (igm[igm.type == 'mean']['x'].values, np.log2(igm[igm.type == 'mean']['y'].values + 1))
data_n = (nk['x'].values, nk['y'].values)

# Retirando as condições iniciais fixas do dicionário, para elas não serem ajustadas
keys_pars = [k for k in pars.keys() if k not in ['Ap0', 'ThN0', 'TkN0', 'B0', 'NK0']]
values_pars = list(pars.values())

# Espaço de busca
bounds = []
for k in keys_pars:
    val = pars[k]
    if val == 0:
        bounds.append((0, 0.1))
    else:
        bounds.append((val * 0.1, val * 10.0))


# Tempo de simulação
tf = 37.0 # dias
dt = 0.01
N = int(tf/dt)
t = np.linspace(0,tf,N)

# FUNÇÃO OBJETIVO
def modelo_objetivo(params):
    p = pars.copy()
    for i, key in enumerate(keys_pars):
        p[key] = params[i]

    sol = solve_ivp(
        modelo, 
        [0, tf], 
        y0, 
        args=(p,), 
        method='Radau', 
        t_eval=t,
        rtol=1e-4,      # Tolerância relativa (um pouco menos rígida que o padrão)
        atol=1e-7,      # Tolerância absoluta
        max_step=0.1)

    if not sol.success or sol.y.shape[1] < len(t):
        return 1e18 
    
    eps = 1e-12

    # Interpolando os passos de tempo
    # sol.y[índice] extrai a linha da variável no tempo simulado 't'
    # data_x[0] são os dias onde temos dados experimentais
    v_sim = np.interp(data_v[0], t, sol.y[0])
    m_sim = np.interp(data_m[0], t, sol.y[12])
    g_sim = np.interp(data_g[0], t, sol.y[13])
    c_sim = np.interp(data_c[0], t, sol.y[14])
    n_sim = np.interp(data_n[0], t, sol.y[15])

    v_log = np.log10(np.maximum(v_sim, 0) + 1)
    m_log = np.log2(np.maximum(m_sim, 0) + 1)
    g_log = np.log2(np.maximum(g_sim, 0) + 1)
    c_log = np.log10(np.maximum(c_sim, 0) + 1) 
    n_log = np.log10(np.maximum(n_sim, 0) + 1) 

    erro_v = np.sum((data_v[1] - v_log)**2)
    erro_m = np.sum((data_m[1] - m_log)**2)
    erro_g = np.sum((data_g[1] - g_log)**2)
    erro_c = np.sum((np.log10(data_c[1] + 1) - c_log)**2)
    erro_n = np.sum((np.log10(data_n[1] + 1) - n_log)**2)

    total_error = erro_v + erro_m + erro_g + erro_c + erro_n

    if np.isnan(total_error):
        return 1e18

    return total_error

def model_adj(params):
    return modelo_objetivo(params)

result = result = differential_evolution(model_adj, 
    bounds, 
    strategy = 'rand1bin', # Mais exploração, menos chance de travar
    popsize = 3, 
    mutation = (0.5, 1.5),  # Aumenta a variabilidade
    recombination = 0.7, 
    tol = 1e-4, 
    disp = True)


# Salvando os melhores parâmetros em .npy
path = r'C:\Users\karla\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\modelo_ajustado'
np.save(path +r'\parametros_otimos.npy', result.x)