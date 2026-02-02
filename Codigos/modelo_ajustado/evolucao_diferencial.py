import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from modelo_covid19_ajustado import modelo, carrega_dados, pars, y0

# Arrumando os dados experimentais -> pega o x e o y dos datasets referente às médias
viremia, il6, igg, igm = carrega_dados()
data_v = (viremia[viremia.type == 'mean']['x'].values, np.log10(viremia[viremia.type == 'mean']['y'].values + 1))
data_c = (il6[il6.type == 'mean']['x'].values, il6[il6.type == 'mean']['y'].values)
data_g = (igg[igg.type == 'mean']['x'].values, np.log2(igg[igg.type == 'mean']['y'].values + 1))
data_m = (igm[igm.type == 'mean']['x'].values, np.log2(igm[igm.type == 'mean']['y'].values + 1))

keys_pars = list(pars.keys())
values_pars = list(pars.values())

# Espaço de busca
bounds = [(val * 0.5, val * 1.5) for val in pars.values()]

# Tempo de simulação
tf = 35.0 #dias
dt = 0.01
N = int(tf/dt)
t = np.linspace(0,tf,N)


# FUNÇÃO OBJETIVO
def modelo_objetivo(params):
    p = pars.copy()

    for i, key in enumerate(keys_pars):
        p[key] = params[i]

    # Condições iniciais fixas
    p['Ap0'] = 1.0e6
    p['ThN0'] = 1.0e6
    p['TkN0'] = 5.0e5
    p['B0'] = 2.5e5
    p['Nmax'] = 5.0e5 


    sol = solve_ivp(modelo, [0, tf], y0, args = (p,), method = 'Radau', t_eval = t)

    # Função auxiliar para calcular o erro
    def calcula_erro(dados_exp, i, log_func):
        t_exp, y_exp = dados_exp
        y_sim = sol.y[i]

        # Interpolação para alinhar os tempos da simulação com os dados
        y_sim_interp = np.interp(t_exp, sol.t, y_sim)
        
        # Aplica transformação logarítmica para bater com os dados
        # Na função calcula_erro, mude para:
        if log_func == 'log10':
            y_sim_interp = np.log10(np.maximum(y_sim_interp, 1e-6)) # maximum evita o zero
        elif log_func == 'log2':
            y_sim_interp = np.log2(np.maximum(y_sim_interp, 1e-6))

        residuo = y_exp - y_sim_interp
        # Erro relativo
        return np.linalg.norm(residuo, 2)/ np.linalg.norm(y_exp, 2)
    erro_viremia = calcula_erro(data_v, 0, 'log10')
    erro_citocina = calcula_erro(data_c, 14, None)
    erro_IgM = calcula_erro(data_m, 12, 'log2')
    erro_IgG = calcula_erro(data_g, 13, 'log2')
    erro_total = erro_viremia + erro_citocina + erro_IgM + erro_IgG

    return erro_total

result = differential_evolution(modelo_objetivo, bounds, strategy='best1bin', 
                                 popsize=3, tol=0.01, disp=True)

# Salvando os melhores parâmetros em JSON
for i in range (len(keys_pars)):
    params_otimos = {keys_pars[i]: result.x[i]}
    with open('parametros_otimos.json', 'w') as f:
        json.dump(params_otimos, f)
    