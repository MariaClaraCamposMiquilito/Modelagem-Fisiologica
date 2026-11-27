import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# O tqdm = biblioteca Python que exibe barras de progresso rápidas e dinâmicas para loops e iteráveis
from scipy.integrate import solve_ivp
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
import os
import pandas as pd

path = "/mnt/c/Users/mique/Documents/Modelagem-Fisiologica/Codigos/Analise_Sensibilidade/Resultados_S1_2511/"

os.makedirs(path, exist_ok=True)

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

l_bound = 0.6
h_bound = 1.4

problem = {
    "num_vars": len(params_ajs),
    "names": params_ajs,
    "bounds": [
        [0.1, 0.9], # beta_ap
        [1, 100], # cap1
        [1e6, 5e7], # cap2
        [pars['gamma_apm'] * l_bound, pars['gamma_apm'] * h_bound], # gamma_apm
        [0.001, 0.01], # beta_apm
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
        [0.001, 0.09], # pi_cNK
    ]
}
n_params = problem["num_vars"]

# ============================================================
# TESTE ISOLADO DA SIMULAÇÃO QUE FALHOU
# ============================================================

# Parâmetros da simulação problemática
p_teste = pars.copy()

p_teste.update({
    'beta_ap': 0.2582258574664593,
    'cap1': 33.39629369508475,
    'cap2': 37298984.04609412,
    'gamma_apm': 0.054730370521545404,
    'beta_apm': 0.009016834106296303,
    'beta_tke': 4.554052868857979e-06,
    'gamma_iNK': 0.00036968555524796243,
    'qn': 0.6448254037201404,
    'Nmax': 2521227.9461324215,
    'dn': 0.05500494609586895,
    'NK0': 143023.7475335598,
    'pi_capm': 253.24491806626318,
    'pi_ci': 0.00599372979092598,
    'pi_ctke': 0.020747927621155973,
    'gamma_c': 912.6112906932831,
    'pi_cNK': 0.00483615565765649
})

# Condições iniciais
y0_teste = list(y0)
y0_teste[15] = p_teste['NK0']


t0 = 0.0
tf = 72.15
n_time = 50

t_eval = np.linspace(t0, tf, n_time)

# Integração
sol_teste = solve_ivp(
        modelo,
        [t0, tf],
        y0_teste,
        t_eval=t_eval,
        args=(p_teste,),
        method="Radau",
        rtol=1e-6,
        atol=1e-9
    )
# Resultado
print("\nRESULTADO DO TESTE:")
print("Sucesso:", sol_teste.success)
print("Mensagem:", sol_teste.message)
print("Último tempo:", sol_teste.t[-1])

if sol_teste.success:
    print("A simulação terminou com sucesso!")
else:
    print("A simulação ainda apresentou problemas.")

nomes = [
    "V", "Ap", "ApM", "I", "ThN", "ThE",
    "TkN", "TkE", "B", "Ps", "Pl", "Bm",
    "IgM", "IgG", "C", "NK"
]

print("\nVALORES MÍNIMOS DAS VARIÁVEIS:")

for i, nome in enumerate(nomes):
    minimo = np.min(sol_teste.y[i])
    print(f"{nome}: {minimo:.6e}")

print("\nTodos os valores são finitos:",
      np.all(np.isfinite(sol_teste.y)))

sol_ref = solve_ivp(
    modelo,
    [t0, tf],
    y0_teste,
    t_eval=t_eval,
    args=(p_teste,),
    method="Radau",
    rtol=1e-8,
    atol=1e-11
)

if not sol_ref.success:
    raise RuntimeError(sol_ref.message)

nomes = [
    "V", "Ap", "ApM", "I", "ThN", "ThE",
    "TkN", "TkE", "B", "Ps", "Pl", "Bm",
    "IgM", "IgG", "C", "NK"
]

print("\nCOMPARAÇÃO DAS SOLUÇÕES:")

for i, nome in enumerate(nomes):
    escala = max(np.max(np.abs(sol_ref.y[i])), 1e-12)

    erro = np.max(
        np.abs(sol_teste.y[i] - sol_ref.y[i])
    ) / escala

    print(f"{nome}: diferença relativa à escala = {erro:.3e}")
import numpy as np

valores = np.array([
    1e-4,
    1e-8,
    1e-12,
    1e-14,
    1e-16,
    1e-18
])

print("COMPARAÇÃO DOS LOGARITMOS\n")

for x in valores:

    log_antigo = np.log10(1 + x)

    log_novo = np.log1p(x) / np.log(10)

    print(f"x = {x:.1e}")
    print(f"  log10(1+x) = {log_antigo:.12e}")
    print(f"  log1p(x)   = {log_novo:.12e}")
    print()

import numpy as np

# Abrir o arquivo
dados = np.load("/mnt/c/Users/mique/Documents/Modelagem-Fisiologica/Codigos/Analise_Sensibilidade/Resultados_S1_40_tol/diagnostico_2e10_ln.npz")

# Mostrar o que existe dentro
print("Conteúdo do arquivo:", dados.files)

# Carregar as matrizes
Y_V = dados["Y_V"]
Y_I = dados["Y_I"]
t_eval = dados["t_eval"]


for nome, Y in [("Viremia", Y_V), ("Infectadas", Y_I)]:

    print(f"\n{nome}")

    for j in [1, 5, 10, 20, 30, 40, 49]:

        valores = Y[:, j]

        n_limite = np.sum(
            np.isclose(valores, np.log1p(1e-12), rtol=0, atol=1e-18)
        )

        porcentagem = 100 * n_limite / len(valores)

        print(
            f"Dia {t_eval[j]:.2f}: "
            f"{porcentagem:.2f}% no limite | "
            f"máximo = {np.max(valores):.6e} | "
            f"variância = {np.var(valores):.6e}"
        )


# Mostrar dimensões
print("\nDimensões:")
print("Y_V:", Y_V.shape)
print("Y_I:", Y_I.shape)
print("t_eval:", t_eval.shape)

# Mostrar algumas informações
print("\nPrimeira simulação de viremia:")
print(Y_V[0, :])

print("\nPrimeira simulação de infectadas:")
print(Y_I[0, :])

print("\nTempos:")
print(t_eval)

print("\n===== INICIANDO COMPARAÇÃO A+B =====")

D = 16
passo = D + 2

for nome, Y in [("Viremia", Y_V), ("Infectadas", Y_I)]:

    A = Y[0::passo, :]
    B = Y[17::passo, :]

    print(f"\n===== {nome} =====")

    for j in [5, 10, 20, 30, 40, 49]:

        valores_AB = np.concatenate((A[:, j], B[:, j]))

        print(f"\nDia {t_eval[j]:.2f}")
        print(f"Variância A+B: {np.var(valores_AB):.6e}")
        print(f"Variância todas: {np.var(Y[:, j]):.6e}")
        print(f"Máximo A+B: {np.max(valores_AB):.6e}")
        print(f"Máximo todas: {np.max(Y[:, j]):.6e}")
