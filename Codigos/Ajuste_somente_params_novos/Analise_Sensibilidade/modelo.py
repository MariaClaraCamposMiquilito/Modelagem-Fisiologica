import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


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
  dTkEdt = p['beta_tk']*(C + 1)*ApM*TkN + p['pi_tk']*ApM*TkE - p['beta_tke']*TkE*V - p['delta_tk']*TkE
  dBdt   = p['alpha_b']*(B0 - B) + p['pi_b1']*V*B + p['pi_b2']*ThE*B - p['beta_ps']*ApM*B - p['beta_pl']*ThE*B - p['beta_bm']*ThE*B
  dPsdt  = p['beta_ps']*ApM*B - p['delta_ps']*Ps
  dPldt  = p['beta_pl']*ThE*B + p['delta_bm']*Bm - p['delta_pl']*Pl 
  dBmdt  = p['beta_bm']*ThE*B + p['pi_bm1']*Bm*(1.0 - (Bm/p['pi_bm2'])) - p['delta_bm']*Bm
  dIgMdt = p['pi_ps']*Ps - p['delta_am']*IgM
  dIgGdt = p['pi_pl']*Pl - p['delta_ag']*IgG
  dCdt   = p['pi_capm']*ApM + p['pi_ci']*I + p['pi_ctke']*TkE + p['pi_cNK']*NK - p['gamma_c']*C
  dNKdt  = p['qn'] * (p['Nmax'] - NK) * I + (p['dn'] * (NK0 - NK))
  return [ dVdt, dApdt, dApMdt, dIdt, dThNdt, dThEdt, dTkNdt, dTkEdt, dBdt, dPsdt, dPldt, dBmdt, dIgMdt, dIgGdt, dCdt, dNKdt ]

# ============================
# Parametros
# ============================
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
#pars['delta_tk']  = 3.0e-1
pars['delta_tk']  = 3.0e-2
#pars['alpha_b']  = 3.58e2
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
pars['qn'] = 0.52             
pars['dn'] = 0.07             
pars['gamma_iNK'] =  0.000574 # Taxa de morte das células Infectadas pelas NK
pars['pi_cNK'] = 0.01         # Taxa de produção de citocinas pelas NK

# Condicoes Iniciais 
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
C0   = 0.0   # pg/mL
NK0  = 1.5e5
Nmax = 3.0e5 

## Condições Iniciais no Pars - Valores Homeostáticos
pars['Ap0']  = Ap0
pars['ThN0'] = ThN0
pars['TkN0'] = TkN0
pars['B0']   = B0
pars['NK0']  = NK0
pars['Nmax'] = Nmax

params_keys = list(pars.keys())

y0 =  [V0, Ap0, ApM0, I0, ThN0, ThE0, TkN0, TkE0, B0, Ps0, Pl0, Bm0, 
       IgM0, IgG0, C0, NK0]

## Tempo da simulação
t0 = 0.0
tf = 40.0
t_eval = np.linspace(t0, tf, 1000)
t_span = (t0, tf)


# ============================
# Solução Base
# ============================

sol = solve_ivp(
   modelo, 
   t_span, 
   y0, 
   args=(pars,), 
   method='Radau', 
   t_eval=t_eval
)

y_base = sol.y

# ============================
# Análise de Sensibilidade
# ============================

def calcula_sensibilidade(delta):
    ## Montando as matrizes zeradas
    S = np.zeros((y_base.shape[0], len(t_eval), len(params_keys)))
    # .shape[0] = tamanho da primeira dimensão do array
    # [16, t_eval, quantidade de parâmetros]

    eps = 1e-12

    # i = indice do parâmetro
    # name = key
    for i, name in enumerate(params_keys):

        print(f"Delta {delta*100}% - parâmetro: {name}")

        # dp = calcula a pertubação
        dp = delta * pars[name]

        pars_plus = pars.copy()
        pars_minus = pars.copy()

        pars_plus[name] += dp
        pars_minus[name] -= dp

        sol_plus = solve_ivp(
            modelo,
            t_span,
            y0,
            args=(pars_plus,),
            method='Radau',
            t_eval=t_eval
        )

        sol_minus = solve_ivp(
            modelo,
            t_span,
            y0,
            args=(pars_minus,),
            method='Radau',
            t_eval=t_eval
        )

        Y_plus = sol_plus.y
        Y_minus = sol_minus.y

        # Diferença Central
        dYdp = (Y_plus - Y_minus) / (2 * dp)

        # Sensibilidade relativa
        for j in range(y_base.shape[0]):
            S[j, :, i] = (pars[name] / (y_base[j] + eps)) * dYdp[j]

    return S


deltas = [0.05, 0.10, 0.20]

sensibilidades = {}

for delta in deltas:
    sensibilidades[delta] = calcula_sensibilidade(delta)

def parametros_mais_influentes(S_variavel, n_top):
    # calcula a média da sensibilidade absoluta ao longo do tempo
    score = np.mean(np.abs(S_variavel), axis=0)

    # pega os 6 últimos elementos, e -1 é o passo, pega de trás pra frente os índices
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
n_top = 7

for nome_var, idx_var in variaveis.items():

    S_ref = sensibilidades[delta_ref]

    indices_top, score = parametros_mais_influentes(S_ref[idx_var], n_top)

    print("\n==============================")
    print(f"Mais influentes em {nome_var}")
    print("==============================")

    for idx in indices_top:
        print(f"{params_keys[idx]} - score = {score[idx]:.4e}")

    fig, ax = plt.subplots(1, 3, figsize=(22, 5), sharey=False)

    for k, delta in enumerate(deltas):

        S_delta = sensibilidades[delta]

        for idx in indices_top:

            #curva = S_delta[idx_var, :, idx]

            #curva_log = np.sign(curva) * np.log10(np.abs(curva) + 1e-12)

            #ax[k].plot(t_eval, curva_log, label = params_keys[idx], linewidth=2)
            ax[k].plot(t_eval, S_delta[idx_var, :, idx], label = params_keys[idx], linewidth=2)

        ax[k].set_title(f'{nome_var} - delta = {delta*100:.0f}%')
        ax[k].set_xlabel('Tempo')
        ax[k].set_ylabel('Sensibilidade relativa')
        ax[k].legend()
        ax[k].grid()

    plt.tight_layout()
    plt.show()