import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# O tqdm = biblioteca Python que exibe barras de progresso rápidas e dinâmicas para loops e iteráveis
from scipy.integrate import solve_ivp
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
import os


def modelo(t, y, p):  

    V, Ap, ApM, I, ThN, ThE, TkN, TkE, B, Ps, Pl, Bm, IgM, IgG, C, NK = y

    Ap0  = p['Ap0']
    ThN0 = p['ThN0']
    TkN0 = p['TkN0']
    B0   = p['B0']
    NK0  = p['NK0']

    dVdt   = p['pi_v']*V - p['kv1']*V*IgG - p['kv1']*V*IgM - p['kv2']*V*TkE - p['kv3']*V*ApM
    dApdt  = p['alpha_ap']*(C + 1.0)*(Ap0 - Ap) - p['beta_ap']*Ap*((p['cap1']*V)/(p['cap2'] + V))
    dApMdt = p['beta_ap']*Ap*((p['cap1']*V)/(p['cap2'] + V)) - p['beta_apm']*ApM*V - p['gamma_apm']*ApM
    dIdt   = p['beta_apm']*ApM*V + p['beta_tke']*TkE*V - p['gamma_apm']*I - p['gamma_iNK']*NK*I
    dThNdt = p['alpha_th']*(ThN0 - ThN) - p['beta_th']*ApM*ThN
    dThEdt = p['beta_th']*ApM*ThN + p['pi_th']*ApM*ThE - p['delta_th']*ThE
    dTkNdt = p['alpha_tk']*(C + 1)*(TkN0 - TkN) - p['beta_tk']*(C + 1)*ApM*TkN
    dTkEdt = p['beta_tk']*(C + 1)*ApM*TkN + p['pi_tk']*ApM*TkE - p['beta_tke']*TkE*V - p['delta_tk']*TkE
    dBdt   = p['alpha_b']*(B0 - B) + p['pi_b1']*V*B + p['pi_b2']*ThE*B - p['beta_ps']*ApM*B - p['beta_pl']*ThE*B - p['beta_bm']*ThE*B
    dPsdt  = p['beta_ps']*ApM*B - p['delta_ps']*Ps
    dPldt  = p['beta_pl']*ThE*B + p['delta_bm']*Bm - p['delta_pl']*Pl 
    dBmdt  = p['beta_bm']*ThE*B + p['pi_bm1']*Bm*(1.0 - (Bm/p['pi_bm2'])) - p['delta_bm']*Bm
    dIgMdt = p['pi_ps']*Ps - p['delta_am']*IgM
    dIgGdt = p['pi_pl']*Pl - p['delta_ag']*IgG
    dCdt   = p['pi_capm']*ApM + p['pi_ci']*I + p['pi_ctke']*TkE - p['gamma_c']*C + p['pi_cNK']*NK
    dNKdt  = p['qn'] * (p['Nmax'] - NK) * I + p['dn'] * (NK0 - NK)

    return [
        dVdt, dApdt, dApMdt, dIdt,
        dThNdt, dThEdt, dTkNdt, dTkEdt,
        dBdt, dPsdt, dPldt, dBmdt,
        dIgMdt, dIgGdt, dCdt, dNKdt
    ]


# ============================================================
# PARÂMETROS
# ============================================================

pars = {}

pars['pi_v']      = 1.47
pars['kv1']       = 9.82e-3
pars['kv2']       = 6.10e-5
pars['kv3']       = 6.45e-2
pars['alpha_ap']  = 1.0
pars['beta_ap']   = 1.79e-1
pars['cap1']      = 8.0
pars['cap2']      = 8.08e6
pars['gamma_apm'] = 4.0e-2
pars['beta_apm']  = 1.33e-2
pars['beta_tke']  = 3.5e-6
pars['alpha_th']  = 2.17e-4
pars['beta_th']   = 1.8e-5
pars['pi_th']     = 1.0e-8
pars['delta_th']  = 3.0e-1
pars['alpha_tk']  = 1.0
pars['beta_tk']   = 1.43e-5
pars['pi_tk']     = 1.0e-8
pars['delta_tk']  = 3.0e-2
pars['alpha_b']   = 3.578236584
pars['pi_b1']     = 8.98e-5
pars['pi_b2']     = 1.27e-8
pars['beta_ps']   = 6.0e-6
pars['beta_pl']   = 5.0e-6
pars['beta_bm']   = 1.0e-6
pars['delta_ps']  = 2.5
pars['delta_pl']  = 3.5e-1
pars['delta_bm']  = 9.75e-4
pars['pi_bm1']    = 1.0e-5
pars['pi_bm2']    = 2.5e3
pars['pi_ps']     = 8.7e-2
pars['pi_pl']     = 1.0e-3
pars['delta_am']  = 7.0e-2
pars['delta_ag']  = 7.0e-2
pars['pi_capm']   = 3.28e2
pars['pi_ci']     = 6.44e-3
pars['pi_ctke']   = 1.78e-2
pars['gamma_c']   = 7.04e2

# parâmetros NK
pars['qn']         = 0.52
pars['dn']         = 0.07
pars['gamma_iNK']  = 0.000574
pars['pi_cNK']     = 0.01


# ============================================================
# CONDIÇÕES INICIAIS
# ============================================================

V0   = 61.0
Ap0  = 1.0e6
ApM0 = 0.0
I0   = 0.0
ThN0 = 1.0e6
ThE0 = 0.0
TkN0 = 5.0e5
TkE0 = 0.0
B0   = 2.5e5
Ps0  = 0.0
Pl0  = 0.0
Bm0  = 0.0
IgM0 = 0.0
IgG0 = 0.0
C0   = 0.0
NK0  = 1.3e5
Nmax = 3.0e6

y0 =  [V0, Ap0, ApM0, I0, ThN0, ThE0, TkN0, TkE0, B0, Ps0, Pl0, 
       Bm0, IgM0, IgG0, C0, NK0]

pars['V0']   = V0
pars['Ap0']  = Ap0
pars['ThN0'] = ThN0
pars['TkN0'] = TkN0
pars['B0']   = B0
pars['NK0']  = NK0
pars['Nmax'] = Nmax

params_keys = list(pars.keys())


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
    path = r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\Analise_Sensibilidade\Resultados_S1_moderado_corrigido"
    plt.savefig(os.path.join(path, f"{title}2.png"))

# ============================================================
# DEFINIÇÃO DO PROBLEMA (SALIB)
# ============================================================
params_sobol = list(pars.keys())

params_ajs = ['pi_v', 'kv3', 'beta_ap', 'cap1', 'cap2', 'beta_apm', 'beta_th', 'delta_th', 'beta_tk',
    'gamma_c', 'dn', 'pi_cNK', 'Ap0', 'TkN0', 'B0']

bounds_de = {
    'pi_v':     [0.8, 1.7],        
    'kv3':      [0.01, 0.10],
    'beta_ap':  [0.001, 0.5],
    'cap1':     [0.001, 5.0],
    'cap2':     [1e6, 6e7],
    'beta_apm': [0.1, 0.9],
    'beta_th':  [0.1, 0.9],
    'delta_th': [0.0005, 0.001],
    'beta_tk':  [3e-5, 2.5e-4],
    'gamma_c':   [0.005, 0.01],
    'dn':       [0.50, 0.95],
    'pi_cNK':   [0.05, 0.5],
    'Ap0':      [1e6, 5e6],
    'TkN0':      [3.5e5, 7e5],
    'B0':     [1.2e5, 4e5] 
}

bounds = []

for name in params_sobol:
    if (name in params_ajs):
        bounds.append(bounds_de[name])
    else:
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
    calc_second_order=False
)

# param_values = conjuntos de parâmetros

print("N_SAMPLES:", N_samples)
print("Número total de simulações:", len(param_values))


# ============================================================
# MALHA TEMPORAL
# ============================================================

t0 = 0.0
tf = 70.0
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

    y0_base = y0.copy()
    y0_base[0] = p['V0']

    sol = solve_ivp(
        modelo,
        [t0, tf],
        y0_base,
        t_eval = t_eval,
        args = (p,),
        method = "Radau"
    )

    if (not sol.success) or (sol.y.shape != (len(y0_base), n_time)) or (not np.all(np.isfinite(sol.y))):
        
        with open(r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\Analise_Sensibilidade\falhas_sobol.txt", "a") as arquivo:
            arquivo.write(f"\n\nFalha na simulacao {i}\n")
            arquivo.write(f"Mensagem: {sol.message}\n")

            for nome in params_sobol:
                linha = f"{nome}: {p[nome]}\n"
                arquivo.write(linha)
        break

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

n_top = 6

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
