import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# O tqdm = biblioteca Python que exibe barras de progresso rápidas e dinâmicas para loops e iteráveis
from scipy.integrate import solve_ivp
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
import os
from itertools import combinations 

# Importa o modelo e as condições iniciais já definidos no seu arquivo
from covid19_modelo_moderado import modelo, y0, pars

# ============================================================
# FUNÇÃO PARA PLOTAR ÍNDICES DE SOBOL + Intervalo de Confiança
# ============================================================

def plot_sobol_top(time, S2, param_names, title, top_pares):

    plt.figure(figsize=(12, 6))

    for i, j in top_pares:
        
        plt.plot(time, 
                 S2[:, i , j], 
                 linewidth=2, 
                 label=f"{param_names[i]} x {param_names[j]}")

    plt.xlabel("Tempo (dias)")
    plt.ylabel("S2")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.ylim(0, 1)
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )
    plt.tight_layout()
    path = r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\Ajuste_somente_params_novos\Ajuste_covid19_moderado\Resultados_S2"
    plt.savefig(os.path.join(path, f"{title}.png"))

## Verificando quaos são os pares mais influentes

def verifica_top_pares(S2, n_top):
    score = np.nanmean(S2, axis = 0);

    ranking = []

    for i, j in combinations(range(score.shape[0]), 2):
        valor = score[i, j]
        if np.isfinite(valor):
            ranking.append((valor, i, j))
    
    ranking.sort(reverse=True)

    top_pares = [(i, j) for valor, i, j in ranking[:n_top]]

    return top_pares, score


# ============================================================
# DEFINIÇÃO DO PROBLEMA (SALIB)
# ============================================================
params_sobol = list(pars.keys())

"""
bounds_de = {
    'pi_v':     [1.60, 1.85],        
    'kv3':      [0.24, 0.35],
    'beta_ap':  [0.16, 0.24],
    'cap1':     [1.00, 1.60],
    'cap2':     [8e6, 2e7],
    'beta_apm': [0.004, 0.007],
    'beta_th':  [1.8e-5, 5e-5],
    'beta_tk':  [4e-4, 8e-4],
    'gama_c':   [700, 1500],
    'Ap0':      [6.5e5, 8.0e5],
    'NK0':      [1.45e5, 1.75e5],
    'Nmax':     [2.5e5, 5.0e5] 
}
"""
bounds = []

#params_ajs = ['pi_v', 'kv3', 'beta_ap', 'cap1', 'cap2', 'beta_apm', 'beta_th', 'beta_tk', 'gama_c', 'Ap0', 'NK0', 'Nmax']

for name in params_sobol:
    bounds.append([pars[name] * 0.5, pars[name] * 1.5])

# O máximo que consegui explorar foi com esses bounds, se eu abrir mais o espaço de busca, o sobol dá erro, pode estar testando combinações absurdas, e dá erro 

problem = {
    "num_vars": len(params_sobol),
    "names": params_sobol,
    "bounds": bounds
}

# ============================================================
# AMOSTRAGEM DE SOBOL
# ============================================================

N_samples = 2**12

# Gerando as amostras
param_values = sobol_sample.sample(
    problem,
    N_samples,
    calc_second_order = True
)

# param_values = conjuntos de parâmetros

print("N_SAMPLES:", N_samples)
print("Número total de simulações:", len(param_values))


# ============================================================
# MALHA TEMPORAL
# ============================================================

t0 = 0.0
tf = 40.0
n_time = 100

t_eval = np.linspace(t0, tf, n_time)

# ============================================================
# EXECUTA TODAS AS SIMULAÇÕES
# ============================================================

# Número total de simulações
n_runs = len(param_values) # tamanho da matriz gerada pelo sobol (número de amostras X número de parâmetros anlisados)

# Matrizes para armazenar resultados
    ## Cada linha = uma simulação
    ## Cada coluna = um instante temporal

Y_V = np.zeros((n_runs, n_time))
Y_IgM = np.zeros((n_runs, n_time))
Y_IgG = np.zeros((n_runs, n_time))
Y_C = np.zeros((n_runs, n_time))
Y_NK = np.zeros((n_runs, n_time))
Y_TCD4 = np.zeros((n_runs, n_time))
Y_TCD8 = np.zeros((n_runs, n_time))
Y_Bcell = np.zeros((n_runs, n_time))

# Loop principal das simulações
## Faz uma simulação para cada conjunto de parâmetros gerado pelo Sobol e armazena as saídas

for i, X in tqdm(enumerate(param_values), total = len(param_values), desc = "Simulações"):
    # Converte vetor de parâmetros em dicionário
    p = pars.copy()

    # Substitui os valores originais por aqueles sorteados pelo Sobol
    for j, name in enumerate(params_sobol):
        p[name] = X[j]

    sol = solve_ivp(
        modelo,
        [t0, tf],
        y0,
        t_eval = t_eval,
        args = (p,),
        method = "Radau"
    )

    Y_V[i, :]     = np.log10(np.clip(sol.y[0], 1e-12, None) + 1)
    Y_IgM[i, :]   = np.log10(np.clip(sol.y[12], 1e-12, None) + 1)
    Y_IgG[i, :]   = np.log10(np.clip(sol.y[13], 1e-12, None) + 1)
    Y_C[i, :]     = np.log10(np.clip(sol.y[14], 1e-12, None) + 1)
    Y_NK[i, :]    = np.log10(np.clip(sol.y[15], 1e-12, None) + 1)
    Y_TCD4[i, :]  = np.log10(np.clip(sol.y[5], 1e-12, None) + 1)
    Y_TCD8[i, :]  = np.log10(np.clip(sol.y[7], 1e-12, None) + 1)
    Y_Bcell[i, :] = np.log10(np.clip(sol.y[8] + sol.y[11], 1e-12, None) + 1)

# ============================================================
# ANÁLISE DE SOBOL AO LONGO DO TEMPO
# ============================================================

n_params = problem["num_vars"]
param_names = problem["names"]

# Matrizes dos índices de Sobol
# Essas matrizes aqui vao armazenar os índices de sensibilidade e não os resultados da simulação
S1_V = np.zeros((n_time, n_params))
ST_V = np.zeros((n_time, n_params))
S2_V = np.zeros((n_time, n_params, n_params))

S1_IgM = np.zeros((n_time, n_params))
ST_IgM = np.zeros((n_time, n_params))
S2_IgM = np.zeros((n_time, n_params, n_params))

S1_IgG = np.zeros((n_time, n_params))
ST_IgG = np.zeros((n_time, n_params))
S2_IgG = np.zeros((n_time, n_params, n_params))

S1_C = np.zeros((n_time, n_params))
ST_C = np.zeros((n_time, n_params))
S2_C = np.zeros((n_time, n_params, n_params))

S1_NK = np.zeros((n_time, n_params))
ST_NK = np.zeros((n_time, n_params))
S2_NK = np.zeros((n_time, n_params, n_params))

S1_TCD4 = np.zeros((n_time, n_params))
ST_TCD4 = np.zeros((n_time, n_params))
S2_TCD4 = np.zeros((n_time, n_params, n_params))

S1_TCD8 = np.zeros((n_time, n_params))
ST_TCD8 = np.zeros((n_time, n_params))
S2_TCD8 = np.zeros((n_time, n_params, n_params))

S1_Bcell = np.zeros((n_time, n_params))
ST_Bcell = np.zeros((n_time, n_params))
S2_Bcell = np.zeros((n_time, n_params, n_params))

print("\nCalculando Sobol temporal...")


# Cálculo dos índices de Sobol

for t_idx in tqdm(range(1, n_time), desc="Calc. Sobol"):

    Si_V = sobol_analyze.analyze(
        problem,
        Y_V[:, t_idx],
        calc_second_order = True,
        print_to_console = False
    )

    S1_V[t_idx, :] = Si_V["S1"]
    ST_V[t_idx, :] = Si_V["ST"]
    S2_V[t_idx, :, :] = Si_V["S2"]


    Si_IgM = sobol_analyze.analyze(
        problem,
        Y_IgM[:, t_idx],
        calc_second_order = True,
        print_to_console = False
    )

    S1_IgM[t_idx, :] = Si_IgM["S1"]
    ST_IgM[t_idx, :] = Si_IgM["ST"]
    S2_IgM[t_idx, :, :] = Si_IgM["S2"]

    Si_IgG = sobol_analyze.analyze(
        problem,
        Y_IgG[:, t_idx],
        calc_second_order = True,
        print_to_console = False
    )

    S1_IgG[t_idx, :] = Si_IgG["S1"]
    ST_IgG[t_idx, :] = Si_IgG["ST"]
    S2_IgG[t_idx, :, :] = Si_IgG["S2"]


    Si_C = sobol_analyze.analyze(
        problem,
        Y_C[:, t_idx],
        calc_second_order = True,
        print_to_console = False
    )

    S1_C[t_idx, :] = Si_C["S1"]
    ST_C[t_idx, :] = Si_C["ST"]
    S2_C[t_idx, :, :] = Si_C["S2"]

    Si_NK = sobol_analyze.analyze(
        problem,
        Y_NK[:, t_idx],
        calc_second_order = True,
        print_to_console = False
    )

    S1_NK[t_idx, :] = Si_NK["S1"]
    ST_NK[t_idx, :] = Si_NK["ST"]
    S2_NK[t_idx, :, :] = Si_NK["S2"]

    Si_TCD4 = sobol_analyze.analyze(
        problem,
        Y_TCD4[:, t_idx],
        calc_second_order = True,
        print_to_console = False
    )

    S1_TCD4[t_idx, :] = Si_TCD4["S1"]
    ST_TCD4[t_idx, :] = Si_TCD4["ST"]
    S2_TCD4[t_idx, :, :] = Si_TCD4["S2"]

    Si_TCD8 = sobol_analyze.analyze(
        problem,
        Y_TCD8[:, t_idx],
        calc_second_order = True,
        print_to_console = False
    )

    S1_TCD8[t_idx, :] = Si_TCD8["S1"]
    ST_TCD8[t_idx, :] = Si_TCD8["ST"]
    S2_TCD8[t_idx, :, :] = Si_TCD8["S2"]
    
    Si_Bcell = sobol_analyze.analyze(
        problem,
        Y_Bcell[:, t_idx],
        calc_second_order = True,
        print_to_console = False
    )

    S1_Bcell[t_idx, :] = Si_Bcell["S1"]
    ST_Bcell[t_idx, :] = Si_Bcell["ST"]
    S2_Bcell[t_idx, :, :] = Si_Bcell["S2"]

# ============================================================
# PLOTS
# ============================================================

n_top = 7

# Viremia
top_V, score_V = verifica_top_pares(S2_V, n_top)
plot_sobol_top(
    t_eval,
    S2_V,
    param_names,
    "Sobol Second-Order - Viremia - Top 7",
    top_V
)

# IgM
top_IgM, score_IgM = verifica_top_pares(S2_IgM, n_top)

plot_sobol_top(
    t_eval,
    S2_IgM,
    param_names,
    "Sobol Second-Order - IgM - Top 7",
    top_IgM
)

# IgG
top_IgG, score_IgG = verifica_top_pares(S2_IgG, n_top)

plot_sobol_top(
    t_eval,
    S2_IgG,
    param_names,
    "Sobol Second-Order - IgG - Top 7",
    top_IgG
)

# Citocinas
top_C, score_C = verifica_top_pares(S2_C, n_top)

plot_sobol_top(
    t_eval,
    S2_C,
    param_names,
    "Sobol Second-Order - Citocinas - Top 7",
    top_C
)

# NK
top_NK, score_NK = verifica_top_pares(S2_NK, n_top)

plot_sobol_top(
    t_eval, 
    S2_NK,
    param_names,
    "Sobol Second-Order - NK - Top 7",
    top_NK
)


# TCD4
top_TCD4, score_TCD4 = verifica_top_pares(S2_TCD4, n_top)

plot_sobol_top(
    t_eval, 
    S2_TCD4,
    param_names,
    "Sobol Second-Order - TCD4 - Top 7",
    top_TCD4
)

# TCD8
top_TCD8, score_TCD8 = verifica_top_pares(S2_TCD8, n_top)

plot_sobol_top(
    t_eval, 
    S2_TCD8,
    param_names,
    "Sobol Second-Order - TCD8 - Top 7",
    top_TCD8
)

# Células B
top_Bcell, score_Bcell = verifica_top_pares(S2_Bcell, n_top)

plot_sobol_top(
    t_eval, 
    S2_Bcell,
    param_names,
    "Sobol Second-Order - Células B - Top 7",
    top_Bcell
)


# Rankings impressos
def imprimir_rankings(nome_saida, top_pares, score, param_names):
    print(f"\nRanking {nome_saida} - S2:")

    for i, j in top_pares:
        print(f"{param_names[i]:12s} x {param_names[j]:12s}\nS2 = {score[i, j]:.4f}")

imprimir_rankings("Viremia", top_V, score_V, param_names)
imprimir_rankings("IgM", top_IgM, score_IgM, param_names)
imprimir_rankings("IgG", top_IgG, score_IgG, param_names)
imprimir_rankings("Citocinas", top_C, score_C, param_names)
imprimir_rankings("NK", top_NK, score_NK, param_names)
imprimir_rankings("TCD4", top_TCD4, score_TCD4, param_names)
imprimir_rankings("TCD8", top_TCD8, score_TCD8, param_names)
imprimir_rankings("Células B", top_Bcell, score_Bcell, param_names)

plt.show()
