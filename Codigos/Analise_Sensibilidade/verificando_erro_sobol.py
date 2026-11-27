
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

t0 = 0.0
tf = 72.15
n_time = 50

t_eval = np.linspace(t0, tf, n_time)

# ============================================================
# DIAGNÓSTICO DA SIMULAÇÃO 1626
# ============================================================
atol = np.array([1e-6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1e-3, 1e-3, 1e-3, 1])
# ordem: V, Ap, ApM, I, ThN, ThE, TkN, TkE, B, Ps, Pl, Bm, IgM, IgG, C, NK

falha = {
    'beta_ap': 0.7805357620120049,
    'cap1': 13.391053657978773,
    'cap2': 9256661.53151542,
    'gamma_apm': 0.0527533905506134,
    'beta_apm': 0.04523698448296637,
    'beta_tke': 2.2590613346546885e-06,
    'gamma_iNK': 0.0007920556574538349,
    'qn': 0.6964117297828197,
    'Nmax': 2830218.717455864,
    'dn': 0.05929116313159466,
    'NK0': 95715.03346413374,
    'pi_capm': 209.80740523934364,
    'pi_ci': 0.003869991023927927,
    'pi_ctke': 0.01719588054433465,
    'gamma_c': 963.7913205623626,
    'pi_cNK': 0.0064432378709316255,
}

p = pars.copy()
p.update(falha)

y0_i = list(y0)
y0_i[15] = p['NK0']
nomes = ['V','Ap','ApM','I','ThN','ThE','TkN','TkE','B','Ps','Pl','Bm','IgM','IgG','C','NK']


for metodo in ("Radau", "LSODA"):
    sol = solve_ivp(
        modelo, [t0, tf], y0_i,
        method=metodo, rtol=1e-8, atol=1e-10,
        args=(p,)
    )
    print(metodo, sol.status, sol.t[-1])
    print("  mínimo de cada variável:", np.round(sol.y.min(axis=1), 4))
    print("  final:", dict(zip(nomes, np.round(sol.y[:, -1], 4))))

print("status:", sol.status)
print("mensagem:", sol.message)
print("último tempo alcançado:", sol.t[-1], "de", tf)
for n, v in zip(nomes, sol.y[:, -1]):
    print(f"{n:>4s} = {v:.4e}")