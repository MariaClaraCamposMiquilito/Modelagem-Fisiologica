import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from covid19_model_reis_2021 import modelo, pars, params_ajs, carrega_dados, y0

# Carregando parâmetros ótimos
params_otimos = np.load(r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\params_otimos\parametros_otimos_2026_04_03.npy")

p = pars.copy()
for i, key in enumerate(params_ajs):
    p[key] = params_otimos[i]
    
nk, viremia, igm, igg, il6 = carrega_dados()

# Rodando o modelo com os parâmetros ótimos
t = np.linspace(0, 37, 3500)
sol = solve_ivp(modelo, [0, 37], y0, args=(p,), method = 'Radau', t_eval = t)

# GRÁFICOS
## Viremia
plt.plot(sol.t, np.log10(sol.y[0] + 1), color='red')
plt.title("Vírus")
plt.xlabel("t (dias)")
plt.ylabel("V")
plt.grid()


x = viremia[viremia.type == 'mean']['x']
y = np.log10(viremia[viremia.type == 'mean']['y'] + 1)

viremia_up = np.log10(viremia[viremia.type == 'up']['y'] + 1)
viremia_down = np.log10(viremia[viremia.type == 'down']['y'] + 1)
viremia_mean = np.log10(viremia[viremia.type == 'mean']['y'] + 1)

y_error = [viremia_mean.to_numpy() - viremia_down.to_numpy(), viremia_up.to_numpy() - viremia_mean.to_numpy()]

plt.errorbar(x,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='red', capsize=4, elinewidth=1)

plt.show()

## Citocinas
plt.plot(sol.t, sol.y[14], color='blue')
plt.title("Citocinas")
plt.xlabel("t (dias)")
plt.ylabel("C")
plt.grid()

x = il6[il6.type == 'mean']['x']
y = il6[il6.type == 'mean']['y']

il6_up = il6[il6.type == 'up']['y']
il6_down = il6[il6.type == 'down']['y']
il6_mean = il6[il6.type == 'mean']['y']

y_error = [il6_mean.to_numpy() - il6_down.to_numpy(), il6_up.to_numpy() - il6_mean.to_numpy()]

plt.errorbar(x,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='blue', capsize=4, elinewidth=1)

plt.show()

## IgG
plt.plot(sol.t, np.log10(sol.y[13] + 1), color='brown')
plt.title("Anticorpos IgG")
plt.xlabel("t (dias)")
plt.ylabel("$I_{gG}$")
plt.grid()


x = igg[igg.type == 'mean']['x']
y = np.log10(igg[igg.type == 'mean']['y'] + 1)

igg_up = np.log10(igg[igg.type == 'up']['y'] + 1)
igg_down = np.log10(igg[igg.type == 'down']['y'] + 1)
igg_mean = np.log10(igg[igg.type == 'mean']['y'] + 1)

y_error = [igg_mean.to_numpy() - igg_down.to_numpy(), igg_up.to_numpy() - igg_mean.to_numpy()]

plt.errorbar(x,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='brown', capsize=4, elinewidth=1)

plt.show()

# IgM
plt.plot(sol.t, np.log10(sol.y[12] + 1), color='orange')
plt.title("Anticorpos IgM")
plt.xlabel("t (dias)")
plt.ylabel("$I_{gM}$")
plt.grid()

x = igm[igm.type == 'mean']['x']
y = np.log10(igm[igm.type == 'mean']['y'] + 1)

igm_up = np.log10(igm[igm.type == 'up']['y'] + 1)
igm_down = np.log10(igm[igm.type == 'down']['y'] + 1)
igm_mean = np.log10(igm[igm.type == 'mean']['y'] + 1)

y_error = [igm_mean.to_numpy() - igm_down.to_numpy(), igm_up.to_numpy() - igm_mean.to_numpy()]

plt.errorbar(x,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='orange', capsize=4, elinewidth=1)

plt.show()

## Natural Killers

plt.scatter(nk['x'].values, np.log10(nk['y'] + 1), label = 'Dados', color = 'green')
plt.grid()
plt.plot(sol.t, np.log10(sol.y[15] + 1), color = 'green')
plt.title("Células Natural Killers")
plt.xlabel("t (dias)")
plt.ylabel("$NK$")
plt.show()
#plt.plot(sol.t, np.log10)

