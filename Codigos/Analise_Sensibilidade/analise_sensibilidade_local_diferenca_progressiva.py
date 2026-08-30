import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
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
# TEMPO
# ============================================================

t0 = 0.0
tf = 70.0
t_eval = np.linspace(t0, tf, 100)
t_span = (t0, tf)

# ============================================================
# SOLUÇÃO BASE
# ============================================================

sol = solve_ivp(
    modelo, 
    t_span, 
    y0, 
    args=(pars,), 
    method='Radau', 
    t_eval=t_eval
)
if not sol.success:
    raise RuntimeError("Erro na solução base: " + sol.message)

t_base = sol.t
y_base = sol.y


# ============================================================
# ANÁLISE DE SENSIBILIDADE LOCAL
# ============================================================

eps = 1e-12

def calcula_sensibilidade(delta):

    S = np.zeros((y_base.shape[0], len(t_eval), len(params_keys)))

    for i, name in enumerate(params_keys):

        print(f"Delta {delta*100:.0f}% - parâmetro: {name}")

        p_pert = pars.copy()
        dp = delta * p_pert[name]

        p_pert[name] += dp

        y0_pert = y0.copy()
        y0_pert[0] = p_pert['V0']

        sol_pert = solve_ivp(
            modelo, 
            t_span, 
            y0, 
            args=(p_pert,), 
            method='Radau', 
            t_eval=t_eval
        )
        if not sol_pert.success :
            print(f"Solver falhou para {name}")
            continue

        y_pert = sol_pert.y

        # Sensibilidade Relativa

        for j in range(y_base.shape[0]):
            S[j, :, i] = (pars[name] / (y_base[j] + eps)) * ((y_pert[j] - y_base[j] ) / dp)

    return S

deltas = [0.05, 0.10, 0.20]

sensibilidades = {}

for delta in deltas:
    sensibilidades[delta] = calcula_sensibilidade(delta)

# ============================================================
# RANKING DOS PARÂMETROS
# ============================================================

def parametros_mais_influentes(S_variavel, n_top):

    score = np.trapezoid(np.abs(S_variavel), t_eval, axis=0)

    indices_top = np.argsort(score)[-n_top:][::-1]

    return indices_top, score


variaveis = {
    'Viremia': 0,
    'Citocinas': 14,
    'IgG': 13,
    'IgM': 12,
    'Natural Killers': 15,
    'TCD4+ efetoras': 5,
    'TCD8+ efetoras': 7,
    'Células B': 8
}


# ============================================================
# GRÁFICOS
# ============================================================

delta_ref = 0.05
n_top = 6

path = r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\Analise_Sensibilidade\Resultados_analise_local"
os.makedirs(path, exist_ok=True)

for nome_var, idx_var in variaveis.items():

    S_ref = sensibilidades[delta_ref]

    indices_top, score = parametros_mais_influentes(S_ref[idx_var], n_top)

    print("\n==============================")
    print(f"Mais influentes em {nome_var}")
    print("==============================")

    for idx in indices_top:
        print(f"{params_keys[idx]} - score = {score[idx]:.4e}")

    fig, ax = plt.subplots(1, 3, figsize=(22, 5), sharey=True)

    for k, delta in enumerate(deltas):

        S_delta = sensibilidades[delta]

        for idx in indices_top:

            curva = S_delta[idx_var, :, idx]

            curva_plot = np.sign(curva) * np.log10(1 + np.abs(curva))

            ax[k].plot(
                t_base,
                S_delta[idx_var, :, idx],
                label=params_keys[idx],
                linewidth=2
            )

        ax[k].set_title(f'{nome_var} - delta = {delta*100:.0f}%')
        ax[k].set_xlabel('Tempo')
        ax[k].set_ylabel(f'{nome_var}')
        ax[k].grid()
        ax[k].legend()

    plt.tight_layout()
    fig.savefig(
        os.path.join(path, f"analise_local_diferenca_progressiva_{nome_var}.png"),
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()
    plt.close(fig)