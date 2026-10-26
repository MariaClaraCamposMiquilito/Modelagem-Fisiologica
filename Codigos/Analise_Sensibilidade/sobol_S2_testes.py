import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # biblioteca Python que exibe barras de progresso rápidas e dinâmicas para loops e iteráveis
from itertools import combinations
from scipy.integrate import solve_ivp
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
import os

# ============================================================
# MODELO
# ============================================================

def modelo(t, y, p):  
  V, Ap, ApM, I, ThN, ThE, TkN, TkE, B, Ps, Pl, Bm, IgM, IgG, C, NK = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14], y[15]

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
  dTkEdt = p['beta_tk']*(C + 1)*ApM*TkN + p['pi_tk']*ApM*ThE - p['beta_tke']*TkE*V - p['delta_tk']*TkE
  dBdt   = p['alpha_b']*(B0 - B) + p['pi_b1']*V*B + p['pi_b2']*ThE*B - p['beta_ps']*ApM*B - p['beta_pl']*ThE*B - p['beta_bm']*ThE*B
  dPsdt  = p['beta_ps']*ApM*B - p['delta_ps']*Ps
  dPldt  = p['beta_pl']*ThE*B + p['delta_bm']*Bm - p['delta_pl']*Pl 
  dBmdt  = p['beta_bm']*ThE*B + p['pi_bm1']*Bm*(1.0 - (Bm/p['pi_bm2'])) - p['delta_bm']*Bm
  dIgMdt = p['pi_ps']*Ps - p['delta_am']*IgM
  dIgGdt = p['pi_pl']*Pl - p['delta_ag']*IgG
  dCdt   = p['pi_capm']*ApM + p['pi_ci']*I + p['pi_ctke']*TkE - p['gamma_c']*C + p['pi_cNK']*NK
  dNKdt  = p['qn'] * (p['Nmax'] - NK) * I + (p['dn'] * (NK0 - NK))

  return [ dVdt, dApdt, dApMdt, dIdt, dThNdt, dThEdt, dTkNdt, dTkEdt, dBdt, dPsdt, dPldt, dBmdt, dIgMdt, dIgGdt, dCdt, dNKdt]
 
# parametros
pars = {}
pars['pi_v']     = 1.47
pars['kv1']      = 9.82e-3
pars['kv2']      = 6.10e-5
pars['kv3']      = 6.45e-2
pars['alpha_ap'] = 1.0
pars['beta_ap']  = 1.79e-1
pars['cap1']     = 8.0
pars['cap2']     = 8.08e6
pars['gamma_apm'] = 4.0e-2
pars['beta_apm'] = 1.33e-2
pars['beta_tke'] = 3.5e-6
pars['alpha_th'] = 2.17e-4
pars['beta_th']  = 1.8e-5
pars['pi_th']    = 1.0e-8
pars['delta_th']  = 3.0e-1
pars['alpha_tk'] = 1.0
pars['beta_tk']  = 1.43e-5
pars['pi_tk']    = 1.0e-8
pars['delta_tk']  = 3.0e-2
pars['alpha_b']  = 3.578236584
pars['pi_b1']    = 8.98e-5
pars['pi_b2']    = 1.27e-8
pars['beta_ps']  = 6.0e-6
pars['beta_pl']  = 5.0e-6
pars['beta_bm']  = 1.0e-6
pars['delta_ps']  = 2.5
pars['delta_pl']  = 3.5e-1
pars['delta_bm']  = 9.75e-4
pars['pi_bm1']   = 1.0e-5
pars['pi_bm2']   = 2.5e3
pars['pi_ps']    = 8.7e-2 
pars['pi_pl']    = 1.0e-3
pars['delta_am']  = 7.0e-2
pars['delta_ag']  = 7.0e-2
pars['pi_capm']  = 3.28e2
pars['pi_ci']    = 6.44e-3
pars['pi_ctke']  = 1.78e-2
pars['gamma_c']   = 7.04e2
pars['qn'] = 0.52             # Taxa de crescimento das células NK
pars['dn'] = 0.07             # Taxa de decaimento natural das Natural Killers
pars['gamma_iNK'] =  0.000574 # Taxa de morte das células Infectadas pelas NK
pars['pi_cNK'] = 0.01         # Taxa de produção de citocinas pelas NK

# condicoes Iniciais 
V0   = 61.0  # copies/mL
Ap0  = 1.0e6 # cells/mL
ApM0 = 0.0   # cells/mL
I0   = 0.0   # cells/mL
ThN0 = 1.0e6 # cells/mL
ThE0 = 0.0   # cells/mL 
TkN0 = 5.0e5 # cells/mL
TkE0 = 0.0   # cells/mL
B0   = 2.5e5 # cells/mL
Ps0  = 0.0   # cells/mL
Pl0  = 0.0   # cells/mL
Bm0  = 0.0   # cells/mL
IgM0 = 0.0   # S/CO
IgG0 = 0.0   # S/CO
C0   = 0.0   # pg/m
NK0  = 1.3e5 
Nmax = 3.0e6
 
# adiciona algumas condicoes iniciais nos parametros
pars['Ap0']  = Ap0
pars['ThN0'] = ThN0
pars['TkN0'] = TkN0
pars['B0']   = B0
pars['NK0']  = NK0
pars['Nmax'] = Nmax

y0 =  [V0, Ap0, ApM0, I0, ThN0, ThE0, TkN0, TkE0, B0, Ps0, Pl0, Bm0, IgM0, IgG0, C0, NK0]

params_ajs = ['beta_ap', 'cap1', 'cap2', 'gamma_apm', 'beta_apm',
    'beta_tke', 'gamma_iNK', 'qn', 'Nmax', 'dn', 'NK0', 'pi_capm',
    'pi_ci', 'pi_ctke', 'gamma_c', 'pi_cNK']

# ============================================================
# PROBLEMA SOBOL
# ============================================================

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

n_samples = 2 ** 16

param_values = sobol_sample.sample(
    problem, 
    n_samples,
    calc_second_order = True
)

n_runs = len(param_values)
print("Numero de amostras: ", n_samples)
print("Numero total de simulações: ", n_runs)

# ============================================================
# MALHA TEMPORAL
# ============================================================

t0 = 0.0
tf = 72.15
n_time = 100

t_eval = np.linspace(t0, tf, n_time)

# ============================================================
# EXECUTA SIMULAÇÕES
# ============================================================

Y_V = np.zeros((n_runs, n_time)) 
Y_ApM = np.zeros((n_runs, n_time)) 
Y_C = np.zeros((n_runs, n_time)) 
Y_I = np.zeros((n_runs, n_time)) 
Y_NK = np.zeros((n_runs, n_time)) 

for i, X in tqdm(enumerate(param_values), total = n_runs, desc = "Simulacoes"):
    
    params = pars.copy()
    params.update(dict(zip(params_ajs, X)))    # sobrescreve só os 16 do Sobol

    y0_i = list(y0)
    y0_i[15] = params['NK0']

    sol = solve_ivp(
        modelo,
        [t0, tf],
        y0_i,
        t_eval = t_eval,
        args = (params,),
        method = 'Radau'
    )

    if (not sol.success) or sol.y.shape[1] != n_time:
        raise RuntimeError(f"Simulação {i} falhou: {sol.message}\nparâmetros: {dict(zip(params_ajs, X))}")

    Y_V[i, :] = sol.y[0]
    Y_ApM[i, :] = sol.y[2]
    Y_C[i, :] = sol.y[14]
    Y_I[i, :] = sol.y[3]
    Y_NK[i, :] = sol.y[15]

# ============================================================
# MATRIZES SOBOL
# ============================================================

S2_V = np.zeros((n_time, n_params, n_params))
S2_ApM = np.zeros((n_time, n_params, n_params))
S2_C = np.zeros((n_time, n_params, n_params))
S2_I = np.zeros((n_time, n_params, n_params))
S2_NK = np.zeros((n_time, n_params, n_params))

# ============================================================
# SOBOL TEMPORAL
# ============================================================
print("\nCalculando Sobol temporal...")

for t_idx in tqdm(range(1, n_time), desc = "Sobol"):
    # Virus
    Si_V = sobol_analyze.analyze(
        problem, 
        Y_V[:, t_idx],
        calc_second_order = True,
        print_to_console = False
    )
    S2_V[t_idx, :, :] = Si_V["S2"]

    # ApM
    Si_ApM = sobol_analyze.analyze(
        problem, 
        Y_ApM[:, t_idx],
        calc_second_order = True,
        print_to_console = False
    )
    S2_ApM[t_idx, :, :] = Si_ApM["S2"]

    # Citocinas
    Si_C = sobol_analyze.analyze(
        problem, 
        Y_C[:, t_idx],
        calc_second_order = True,
        print_to_console = False
    )
    S2_C[t_idx, :, :] = Si_C["S2"]

    # Células Infectadas
    Si_I = sobol_analyze.analyze(
        problem, 
        Y_I[:, t_idx],
        calc_second_order = True,
        print_to_console = False
    )
    S2_I[t_idx, :, :] = Si_I["S2"]

    # Natural Killers
    Si_NK = sobol_analyze.analyze(
        problem, 
        Y_NK[:, t_idx],
        calc_second_order = True,
        print_to_console = False
    )
    S2_NK[t_idx, :, :] = Si_NK["S2"]

# ============================================================
# PLOT S2
# ============================================================

def plot_second_order(time, S2, params_names, title, top):
    pares = list(combinations(range(len(params_names)), 2))

    valores = []
    for i, j in pares:
        serie_temporal = S2[:, i, j]          # S2 do par (i, j) em todos os instantes
        s2_max = np.nanmax(serie_temporal)    # maior valor daquele par
        valores.append((s2_max, i, j))

    valores.sort(reverse=True)
    valores = valores[:top]

    plt.figure(figsize = (12, 7))

    for s2_max, i, j in valores:
        plt.plot(time, S2[:, i, j], label = f"{params_names[i]} X {params_names[j]}")
    plt.xlabel("Tempo")
    plt.ylabel("S2")
    plt.title(title)
    plt.grid()
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )
    plt.tight_layout()

    pasta_destino = "/home/mariamiquelito/Modelagem-Fisiologica/Codigos/Analise_Sensibilidade"
    
    os.makedirs(pasta_destino, exist_ok=True)
    
    caminho_completo = os.path.join(pasta_destino, title + ".png")
    plt.savefig(caminho_completo, dpi=300, bbox_inches="tight")
    
# ============================================================
# PLOTS
# ============================================================
top = 10
plot_second_order(t_eval, S2_V, params_ajs, "Sobol_Segunda_Ordem_Virus_2e16", top)
plot_second_order(t_eval, S2_ApM, params_ajs, "Sobol_Segunda_Ordem_APM_2e16", top)
plot_second_order(t_eval, S2_C, params_ajs, "Sobol_Segunda_Ordem_Citocinas_2e16", top)
plot_second_order(t_eval, S2_I, params_ajs, "Sobol_Segunda_Ordem_Infectadas_2e16", top)
plot_second_order(t_eval, S2_NK, params_ajs, "Sobol_Segunda_Ordem_NK_2e16", top)

# ============================================================
# RANKING DAS INTERAÇÕES
# ============================================================
def ranking_pares(S2, top):
    ranking = []
    for i, j in combinations(range(n_params), 2):
        valor = np.nanmax(S2[:, i, j])

        ranking.append((
            valor, 
            params_ajs[i],
            params_ajs[j]
        ))
    ranking.sort(reverse = True) # Ordem Decrescente
    ranking = ranking[:top]            # fica só com os top N

    for valor, p1, p2, in ranking:
        print(f"{p1:12s} x {p2:12s} S2_max = {valor:.4f}")

print("\n==============================")
print("VÍRUS")