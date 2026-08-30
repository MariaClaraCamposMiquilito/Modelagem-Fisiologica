import numpy as np
import pandas as pd

def modelo(t, y, p):  
  #pi_v, k_v1, k_v2, k_v3, alfa_ap, beta_ap, c_ap1, c_ap2, delta_apm, beta_apm, beta_tke, alfa_th, beta_th, pi_th, delta_th, alfa_tk, beta_tk, pi_tk, delta_tk, alfa_b,
            #pi_b1, pi_b2, beta_ps, beta_pl, beta_bm, delta_ps, delta_pl, gamma_bm, pi_bm1, pi_bm2, pi_ps, pi_pl, delta_am, delta_ag, pi_c_apm, pi_c_i, pi_c_tke, delta_c, qn, Nmax, 
            #dn, gamma_iNK, pi_cNK, Ap0, Thn0, Tkn0, B0, NK0
  
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
  dCdt   = p['pi_capm']*ApM + p['pi_ci']*I + p['pi_ctke']*TkE - p['gamma_c']*C + p['pi_cNK']*NK
  dNKdt  = p['qn'] * (p['Nmax'] - NK) * I + (p['dn'] * (NK0 - NK))

  return [ dVdt, dApdt, dApMdt, dIdt, dThNdt, dThEdt, dTkNdt, dTkEdt, dBdt, dPsdt, dPldt, dBmdt, dIgMdt, dIgGdt, dCdt, dNKdt]

# Extraindo os dados exp erimentais
def carrega_dados():  
  data_nk = pd.read_csv(r'C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Data\covid_moderado\NK_covid_moderado.csv', sep = ',')
  data_viremia = pd.read_csv(r'C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Data\viremia_covid.csv', sep = ',')
  data_igm = pd.read_csv(r'C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Data\covid_moderado\IgM_covid_moderado.csv', sep = ',')
  data_igg = pd.read_csv(r'C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Data\covid_moderado\IgG_covid_moderado.csv', sep = ',')
  data_il6 = pd.read_csv(r'C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Data\citocinas_covid.csv', sep = ',')
  data_tcd4 = pd.read_csv(r'C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Data\covid_moderado\TCD4_covid_moderado.csv', sep = ',')
  data_tcd8 = pd.read_csv(r'C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Data\covid_moderado\TCD8_covid_moderado.csv', sep = ',')
  data_cellB = pd.read_csv(r'C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Data\covid_moderado\Bcell_covid_moderado.csv', sep = ',')

  data_nk['y'] = data_nk['y'] / (10**3)
  data_igm['y'] = data_igm['y'] / (10**3)
  data_igg['y'] = data_igg['y'] / (10**3)
  data_tcd4['y'] = data_tcd4['y'] / (10**3)
  data_tcd8['y'] = data_tcd8['y'] / (10**3)
  data_cellB['y'] = data_cellB['y'] / (10**3)

  return data_nk, data_viremia, data_igm, data_igg, data_il6, data_tcd4, data_tcd8, data_cellB

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

params_ajs = ['pi_v', 'kv3', 'beta_ap', 'cap1', 'cap2', 'beta_apm', 'beta_th', 'delta_th', 'beta_tk',
    'gamma_c', 'dn', 'pi_cNK', 'Ap0', 'TkN0', 'B0']