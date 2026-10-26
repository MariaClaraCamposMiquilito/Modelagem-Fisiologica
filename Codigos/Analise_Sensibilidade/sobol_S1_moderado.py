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
pars['Nmax']       = 3.0e6


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


y0 =  [V0, Ap0, ApM0, I0, ThN0, ThE0, TkN0, TkE0, B0, Ps0, Pl0, 
       Bm0, IgM0, IgG0, C0, NK0]

pars['V0']   = V0
pars['Ap0']  = Ap0
pars['ThN0'] = ThN0
pars['TkN0'] = TkN0
pars['B0']   = B0
pars['NK0']  = NK0


params_keys = list(pars.keys())

# ============================================================
# PROBLEMA SOBOL
# ============================================================
params_ajs = ['beta_ap', 'cap1', 'cap2', 'gamma_apm', 'beta_apm',
    'beta_tke', 'gamma_iNK', 'qn', 'Nmax', 'dn', 'NK0', 'pi_capm',
    'pi_ci', 'pi_ctke', 'gamma_c', 'pi_cNK']

l_bound = 0.9
h_bound = 1.1

problem = {
    "num_vars": len(params_ajs),
    "names": params_ajs,
    "bounds": [
        [pars['beta_ap'] * l_bound, pars['beta_ap'] * h_bound], # beta_ap
        [8, 11], # cap1
        [7e6, 1e7], # cap2
        [pars['gamma_apm'] * l_bound, pars['gamma_apm'] * h_bound], # gamma_apm
        [pars['beta_apm'] * l_bound, pars['beta_apm'] * h_bound], # beta_apm
        [pars['beta_tke'] * l_bound, pars['beta_tke'] * h_bound], # beta_tke
        [pars['gamma_iNK'] * l_bound, pars['gamma_iNK'] * h_bound], # gamma_iNK
        [pars['qn'] * l_bound, pars['qn'] * h_bound], # qn
        [pars['Nmax'] * l_bound, pars['Nmax'] * h_bound], # Nmax
        [0.05, 0.1], # dn
        [pars['NK0'] * l_bound, pars['NK0'] * h_bound], # NK0
        [pars['pi_capm'] * l_bound, pars['pi_capm'] * h_bound], # pi_capm
        [pars['pi_ci'] * l_bound, pars['pi_ci'] * h_bound], # pi_ci
        [pars['pi_ctke'] * l_bound, pars['pi_ctke'] * h_bound], # pi_ctke
        [pars['gamma_c'] * l_bound, pars['gamma_c'] * h_bound], # gamma_c
        [0.001, 0.003], # pi_cNK
    ]
}

n_params = problem["num_vars"]

# ============================================================
# AMOSTRAGEM
# ============================================================

N_samples = 2**12

# Gerando as amostras
param_values = sobol_sample.sample(
    problem,
    N_samples,
    calc_second_order=False
)

n_runs = len(param_values)


# param_values = conjuntos de parâmetros
print("N_SAMPLES:", N_samples)
print("Número total de simulações:", n_runs)


# ============================================================
# MALHA TEMPORAL
# ============================================================

t0 = 0.0
tf = 72.15
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
Y_C = np.zeros((n_runs, n_time))
Y_NK = np.zeros((n_runs, n_time))
Y_ApM = np.zeros((n_runs, n_time))
Y_I = np.zeros((n_runs, n_time))

# Loop principal das simulações
## Faz uma simulação para cada conjunto de parâmetros gerado pelo Sobol e armazena as saídas

for i, X in tqdm(enumerate(param_values), total = len(param_values), desc = "Simulações"):
    # Converte vetor de parâmetros em dicionário
    p = pars.copy()

    p.update(dict(zip(params_ajs, X)))    # sobrescreve só os 16 do Sobol

    y0_i = list(y0)
    y0_i[15] = p['NK0']

    sol = solve_ivp(
        modelo,
        [t0, tf],
        y0_i,
        t_eval = t_eval,
        args = (p,),
        method = "Radau"
    )
    if (not sol.success) or sol.y.shape[1] != n_time:
        raise RuntimeError(f"Simulação {i} falhou: {sol.message}\nparâmetros: {dict(zip(params_ajs, X))}")

    Y_V[i, :]   = np.log10(np.clip(sol.y[0], 1e-12, None) + 1)
    Y_C[i, :]   = np.log10(np.clip(sol.y[14], 1e-12, None) + 1)
    Y_NK[i, :]  = np.log10(np.clip(sol.y[15], 1e-12, None) + 1)
    Y_ApM[i, :] = np.log10(np.clip(sol.y[2], 1e-12, None) + 1)
    Y_I[i, :]   = np.log10(np.clip(sol.y[3], 1e-12, None) + 1)

# ============================================================
# MATRIZES SOBOL
# ============================================================
param_names = problem["names"]

# Matrizes dos índices de Sobol
# Essas matrizes aqui vao armazenar os índices de sensibilidade e não os resultados da simulação
S1_V = np.zeros((n_time, n_params))
ST_V = np.zeros((n_time, n_params))

S1_C = np.zeros((n_time, n_params))
ST_C = np.zeros((n_time, n_params))

S1_NK = np.zeros((n_time, n_params))
ST_NK = np.zeros((n_time, n_params))

S1_ApM = np.zeros((n_time, n_params))
ST_ApM = np.zeros((n_time, n_params))

S1_I = np.zeros((n_time, n_params))
ST_I = np.zeros((n_time, n_params))

# Matrizes dos intervalos de confiança
S1_conf_V = np.zeros((n_time, n_params))
ST_conf_V = np.zeros((n_time, n_params))

S1_conf_C = np.zeros((n_time, n_params))
ST_conf_C = np.zeros((n_time, n_params))

S1_conf_NK = np.zeros((n_time, n_params))
ST_conf_NK = np.zeros((n_time, n_params))

S1_conf_ApM = np.zeros((n_time, n_params))
ST_conf_ApM = np.zeros((n_time, n_params))

S1_conf_I = np.zeros((n_time, n_params))
ST_conf_I = np.zeros((n_time, n_params))


# ============================================================
# ANÁLISE DE SOBOL AO LONGO DO TEMPO
# ============================================================

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


    Si_ApM= sobol_analyze.analyze(
        problem,
        Y_ApM[:, t_idx],
        calc_second_order=False,
        print_to_console=False
    )

    S1_ApM[t_idx, :] = Si_ApM["S1"]
    ST_ApM[t_idx, :] = Si_ApM["ST"]
    S1_conf_ApM[t_idx, :] = Si_ApM["S1_conf"]
    ST_conf_ApM[t_idx, :] = Si_ApM["ST_conf"]

    Si_I = sobol_analyze.analyze(
        problem,
        Y_I[:, t_idx],
        calc_second_order=False,
        print_to_console=False
    )

    S1_I[t_idx, :] = Si_I["S1"]
    ST_I[t_idx, :] = Si_I["ST"]
    S1_conf_I[t_idx, :] = Si_I["S1_conf"]
    ST_conf_I[t_idx, :] = Si_I["ST_conf"]

    
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
    plt.legend()
    plt.tight_layout()
    path = r"/mnt/c/Users/mique/OneDrive/Documentos/UFJF/Modelagem Fisiologica/Codigos/Analise_Sensibilidade/Resultados_S1_testes"
    plt.savefig(os.path.join(path, f"{title}.png"))

    plt.close()
# ============================================================
# PLOTS
# ============================================================

n_top = 10

# Viremia
score_V = np.nanmax(ST_V, axis=0)
top_V = np.argsort(score_V)[-n_top:][::-1]

plot_sobol_top_with_ci(
    t_eval,
    ST_V,
    ST_conf_V,
    "Sobol Total-Order - Viremia - Top 10",
    "ST",
    param_names,
    top_V
)

plot_sobol_top_with_ci(
    t_eval,
    S1_V,
    S1_conf_V,
    "Sobol First-Order - Viremia - Top 10",
    "S1",
    param_names,
    top_V
)

# Citocinas
score_C = np.nanmax(ST_C, axis=0)
top_C = np.argsort(score_C)[-n_top:][::-1]

plot_sobol_top_with_ci(
    t_eval, ST_C, ST_conf_C,
    "Sobol Total-Order - Citocinas - Top 10",
    "ST", param_names, top_C
)

plot_sobol_top_with_ci(
    t_eval, S1_C, S1_conf_C,
    "Sobol First-Order - Citocinas - Top 10",
    "S1", param_names, top_C
)

# NK
score_NK = np.nanmax(ST_NK, axis=0)
top_NK = np.argsort(score_NK)[-n_top:][::-1]

plot_sobol_top_with_ci(
    t_eval, ST_NK, ST_conf_NK,
    "Sobol Total-Order - NK - Top 10",
    "ST", param_names, top_NK
)

plot_sobol_top_with_ci(
    t_eval, S1_NK, S1_conf_NK,
    "Sobol First-Order - NK - Top 10",
    "S1", param_names, top_NK
)

# Células ApM
score_ApM = np.nanmax(ST_ApM, axis=0)
top_ApM = np.argsort(score_ApM)[-n_top:][::-1]

plot_sobol_top_with_ci(
    t_eval, ST_ApM, ST_conf_ApM,
    "Sobol Total-Order - ApM - Top 10",
    "ST", param_names, top_ApM
)

plot_sobol_top_with_ci(
    t_eval, S1_ApM, S1_conf_ApM,
    "Sobol First-Order - ApM - Top 10",
    "S1", param_names, top_ApM
)

# Células Infectadas
score_I = np.nanmax(ST_I, axis=0)
top_I = np.argsort(score_I)[-n_top:][::-1]

plot_sobol_top_with_ci(
    t_eval, ST_I, ST_conf_I,
    "Sobol Total-Order - I - Top 10",
    "ST", param_names, top_I
)

plot_sobol_top_with_ci(
    t_eval, S1_I, S1_conf_I,
    "Sobol First-Order - I - Top 10",
    "S1", param_names, top_I
)

# Rankings impressos
print("\nRanking Viremia - ST:")
for idx in top_V:
    print(param_names[idx], "ST =", score_V[idx])

print("\nRanking Citocinas - ST:")
for idx in top_C:
    print(param_names[idx], "ST =", score_C[idx])

print("\nRanking NK - ST:")
for idx in top_NK:
    print(param_names[idx], "ST =", score_NK[idx])

print("\nRanking ApM - ST :")
for idx in top_ApM:
    print(param_names[idx], "ST =", score_ApM[idx])

print("\nRanking I - ST:")
for idx in top_I:
    print(param_names[idx], "ST =", score_I[idx])

plt.show()