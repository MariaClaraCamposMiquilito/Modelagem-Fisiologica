import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# O tqdm = biblioteca Python que exibe barras de progresso rápidas e dinâmicas para loops e iteráveis
from scipy.integrate import solve_ivp
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
import os

# Importa o modelo e as condições iniciais já definidos no seu arquivo
from covid19_modelo_moderado import modelo, y0, pars

# ============================================================
# FUNÇÃO PARA PLOTAR ÍNDICES DE SOBOL + Intervalo de Confiança
# ============================================================

def plot_sobol_top_with_ci(time,
                           indices,
                           conf,
                           title,
                           ylabel,
                           param_names,
                           top_idx):
    """
    Plota somente os parâmetros mais sensíveis,
    com intervalo de confiança.
    """

    plt.figure(figsize=(12, 6))

    for i in top_idx:
        # Média dos índices de Sobol
        mean = indices[:, i]
        # Limites inferior e superior do IC
        lower = mean - conf[:, i]
        upper = mean + conf[:, i]

        # Curva
        plt.plot(time, mean, linewidth=2, label=param_names[i])
        # Região sombreada
        plt.fill_between(time, lower, upper, alpha=0.2)

    plt.xlabel("Tempo (dias)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    path = r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\Ajuste_somente_params_novos\Ajuste_covid19_moderado\Resultados_SOBOL"
    plt.savefig(os.path.join(path, f"{title}2.png"))

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

N_samples = 2**13

# Gerando as amostras
param_values = sobol_sample.sample(
    problem,
    N_samples,
    calc_second_order=False
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

S1_IgM = np.zeros((n_time, n_params))
ST_IgM = np.zeros((n_time, n_params))

S1_IgG = np.zeros((n_time, n_params))
ST_IgG = np.zeros((n_time, n_params))

S1_C = np.zeros((n_time, n_params))
ST_C = np.zeros((n_time, n_params))

S1_NK = np.zeros((n_time, n_params))
ST_NK = np.zeros((n_time, n_params))

S1_TCD4 = np.zeros((n_time, n_params))
ST_TCD4 = np.zeros((n_time, n_params))

S1_TCD8 = np.zeros((n_time, n_params))
ST_TCD8 = np.zeros((n_time, n_params))

S1_Bcell = np.zeros((n_time, n_params))
ST_Bcell = np.zeros((n_time, n_params))

# Matrizes dos intervalos de confiança

S1_conf_V = np.zeros((n_time, n_params))
ST_conf_V = np.zeros((n_time, n_params))

S1_conf_IgM = np.zeros((n_time, n_params))
ST_conf_IgM = np.zeros((n_time, n_params))

S1_conf_IgG = np.zeros((n_time, n_params))
ST_conf_IgG = np.zeros((n_time, n_params))

S1_conf_C = np.zeros((n_time, n_params))
ST_conf_C = np.zeros((n_time, n_params))

S1_conf_NK = np.zeros((n_time, n_params))
ST_conf_NK = np.zeros((n_time, n_params))

S1_conf_TCD4 = np.zeros((n_time, n_params))
ST_conf_TCD4 = np.zeros((n_time, n_params))

S1_conf_TCD8 = np.zeros((n_time, n_params))
ST_conf_TCD8 = np.zeros((n_time, n_params))

S1_conf_Bcell = np.zeros((n_time, n_params))
ST_conf_Bcell = np.zeros((n_time, n_params))

print("\nCalculando Sobol temporal...")


# Cálculo dos índices de Sobol

for t_idx in tqdm(range(1, n_time), desc="Calc. Sobol"):

    Si_V = sobol_analyze.analyze(
        problem,
        Y_V[:, t_idx],
        calc_second_order=False,
        print_to_console=False
    )

    S1_V[t_idx, :] = Si_V["S1"]
    ST_V[t_idx, :] = Si_V["ST"]
    S1_conf_V[t_idx, :] = Si_V["S1_conf"]
    ST_conf_V[t_idx, :] = Si_V["ST_conf"]

    Si_IgM = sobol_analyze.analyze(
        problem,
        Y_IgM[:, t_idx],
        calc_second_order=False,
        print_to_console=False
    )

    S1_IgM[t_idx, :] = Si_IgM["S1"]
    ST_IgM[t_idx, :] = Si_IgM["ST"]
    S1_conf_IgM[t_idx, :] = Si_IgM["S1_conf"]
    ST_conf_IgM[t_idx, :] = Si_IgM["ST_conf"]

    Si_IgG = sobol_analyze.analyze(
        problem,
        Y_IgG[:, t_idx],
        calc_second_order=False,
        print_to_console=False
    )

    S1_IgG[t_idx, :] = Si_IgG["S1"]
    ST_IgG[t_idx, :] = Si_IgG["ST"]
    S1_conf_IgG[t_idx, :] = Si_IgG["S1_conf"]
    ST_conf_IgG[t_idx, :] = Si_IgG["ST_conf"]

    Si_C = sobol_analyze.analyze(
        problem,
        Y_C[:, t_idx],
        calc_second_order=False,
        print_to_console=False
    )

    S1_C[t_idx, :] = Si_C["S1"]
    ST_C[t_idx, :] = Si_C["ST"]
    S1_conf_C[t_idx, :] = Si_C["S1_conf"]
    ST_conf_C[t_idx, :] = Si_C["ST_conf"]

    Si_NK = sobol_analyze.analyze(
        problem,
        Y_NK[:, t_idx],
        calc_second_order=False,
        print_to_console=False
    )

    S1_NK[t_idx, :] = Si_NK["S1"]
    ST_NK[t_idx, :] = Si_NK["ST"]
    S1_conf_NK[t_idx, :] = Si_NK["S1_conf"]
    ST_conf_NK[t_idx, :] = Si_NK["ST_conf"]

    Si_TCD4 = sobol_analyze.analyze(
        problem,
        Y_TCD4[:, t_idx],
        calc_second_order=False,
        print_to_console=False
    )

    S1_TCD4[t_idx, :] = Si_TCD4["S1"]
    ST_TCD4[t_idx, :] = Si_TCD4["ST"]
    S1_conf_TCD4[t_idx, :] = Si_TCD4["S1_conf"]
    ST_conf_TCD4[t_idx, :] = Si_TCD4["ST_conf"]


    Si_TCD8 = sobol_analyze.analyze(
        problem,
        Y_TCD8[:, t_idx],
        calc_second_order=False,
        print_to_console=False
    )

    S1_TCD8[t_idx, :] = Si_TCD8["S1"]
    ST_TCD8[t_idx, :] = Si_TCD8["ST"]
    S1_conf_TCD8[t_idx, :] = Si_TCD8["S1_conf"]
    ST_conf_TCD8[t_idx, :] = Si_TCD8["ST_conf"]

    Si_Bcell = sobol_analyze.analyze(
        problem,
        Y_Bcell[:, t_idx],
        calc_second_order=False,
        print_to_console=False
    )

    S1_Bcell[t_idx, :] = Si_Bcell["S1"]
    ST_Bcell[t_idx, :] = Si_Bcell["ST"]
    S1_conf_Bcell[t_idx, :] = Si_Bcell["S1_conf"]
    ST_conf_Bcell[t_idx, :] = Si_Bcell["ST_conf"]

# ============================================================
# PLOTS
# ============================================================

n_top = 7

# Viremia
score_V = np.nanmean(ST_V, axis=0)
top_V = np.argsort(score_V)[-n_top:][::-1]

plot_sobol_top_with_ci(
    t_eval,
    ST_V,
    ST_conf_V,
    "Sobol Total-Order - Viremia - Top 6",
    "ST",
    param_names,
    top_V
)

plot_sobol_top_with_ci(
    t_eval,
    S1_V,
    S1_conf_V,
    "Sobol First-Order - Viremia - Top 6",
    "S1",
    param_names,
    top_V
)

# IgM
score_IgM = np.nanmean(ST_IgM, axis=0)
top_IgM = np.argsort(score_IgM)[-n_top:][::-1]

plot_sobol_top_with_ci(
    t_eval,
    ST_IgM,
    ST_conf_IgM,
    "Sobol Total-Order - IgM - Top 6",
    "ST",
    param_names,
    top_IgM
)

plot_sobol_top_with_ci(
    t_eval,
    S1_IgM,
    S1_conf_IgM,
    "Sobol First-Order - IgM - Top 6",
    "S1",
    param_names,
    top_IgM
)

# IgG
score_IgG = np.nanmean(ST_IgG, axis=0)
top_IgG = np.argsort(score_IgG)[-n_top:][::-1]

plot_sobol_top_with_ci(
    t_eval,
    ST_IgG,
    ST_conf_IgG,
    "Sobol Total-Order - IgG - Top 6",
    "ST",
    param_names,
    top_IgG
)

plot_sobol_top_with_ci(
    t_eval,
    S1_IgG,
    S1_conf_IgG,
    "Sobol First-Order - IgG - Top 6",
    "S1",
    param_names,
    top_IgG
)
# Citocinas
score_C = np.nanmean(ST_C, axis=0)
top_C = np.argsort(score_C)[-n_top:][::-1]

plot_sobol_top_with_ci(
    t_eval, ST_C, ST_conf_C,
    "Sobol Total-Order - Citocinas - Top 6",
    "ST", param_names, top_C
)

plot_sobol_top_with_ci(
    t_eval, S1_C, S1_conf_C,
    "Sobol First-Order - Citocinas - Top 6",
    "S1", param_names, top_C
)

# NK
score_NK = np.nanmean(ST_NK, axis=0)
top_NK = np.argsort(score_NK)[-n_top:][::-1]

plot_sobol_top_with_ci(
    t_eval, ST_NK, ST_conf_NK,
    "Sobol Total-Order - NK - Top 6",
    "ST", param_names, top_NK
)

plot_sobol_top_with_ci(
    t_eval, S1_NK, S1_conf_NK,
    "Sobol First-Order - NK - Top 6",
    "S1", param_names, top_NK
)

# TCD4
score_TCD4 = np.nanmean(ST_TCD4, axis=0)
top_TCD4 = np.argsort(score_TCD4)[-n_top:][::-1]

plot_sobol_top_with_ci(
    t_eval, ST_TCD4, ST_conf_TCD4,
    "Sobol Total-Order - TCD4 - Top 6",
    "ST", param_names, top_TCD4
)

plot_sobol_top_with_ci(
    t_eval, S1_TCD4, S1_conf_TCD4,
    "Sobol First-Order - TCD4 - Top 6",
    "S1", param_names, top_TCD4
)

# TCD8
score_TCD8 = np.nanmean(ST_TCD8, axis=0)
top_TCD8 = np.argsort(score_TCD8)[-n_top:][::-1]

plot_sobol_top_with_ci(
    t_eval, ST_TCD8, ST_conf_TCD8,
    "Sobol Total-Order - TCD8 - Top 6",
    "ST", param_names, top_TCD8
)

plot_sobol_top_with_ci(
    t_eval, S1_TCD8, S1_conf_TCD8,
    "Sobol First-Order - TCD8 - Top 6",
    "S1", param_names, top_TCD8
)

# Células B
score_Bcell = np.nanmean(ST_Bcell, axis=0)
top_Bcell = np.argsort(score_Bcell)[-n_top:][::-1]

plot_sobol_top_with_ci(
    t_eval, ST_Bcell, ST_conf_Bcell,
    "Sobol Total-Order - Células B - Top 6",
    "ST", param_names, top_Bcell
)

plot_sobol_top_with_ci(
    t_eval, S1_Bcell, S1_conf_Bcell,
    "Sobol First-Order - Células B - Top 6",
    "S1", param_names, top_Bcell
)

# Rankings impressos
print("\nRanking Viremia - ST médio:")
for idx in top_V:
    print(param_names[idx], "ST_medio =", score_V[idx])

print("\nRanking IgM - ST médio:")
for idx in top_IgM:
    print(param_names[idx], "ST_medio =", score_IgM[idx])

print("\nRanking IgG - ST médio:")
for idx in top_IgG:
    print(param_names[idx], "ST_medio =", score_IgG[idx])

print("\nRanking Citocinas - ST médio:")
for idx in top_C:
    print(param_names[idx], "ST_medio =", score_C[idx])

print("\nRanking NK - ST médio:")
for idx in top_NK:
    print(param_names[idx], "ST_medio =", score_NK[idx])

print("\nRanking TCD4 - ST médio:")
for idx in top_TCD4:
    print(param_names[idx], "ST_medio =", score_TCD4[idx])

print("\nRanking TCD8 - ST médio:")
for idx in top_TCD8:
    print(param_names[idx], "ST_medio =", score_TCD8[idx])

print("\nRanking Células B - ST médio:")
for idx in top_Bcell:
    print(param_names[idx], "ST_medio =", score_Bcell[idx])

plt.show()
