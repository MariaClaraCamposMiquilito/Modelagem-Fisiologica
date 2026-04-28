import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from covid19_model_severo import modelo, y0, pars, params_ajs, carrega_dados

nk, viremia, igm, igg, il6, tcd4, tcd8, cellB = carrega_dados()

# Carregando parâmetros ótimos
params_otimos = np.load(r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\params_otimos\parametros_otimos_moderado02.npy")
 
p = pars.copy()
for i, key in enumerate(params_ajs):
    p[key] = params_otimos[i]
    

# Rodando o modelo com os parâmetros ótimos
t = np.linspace(0, 39, 3500)
sol = solve_ivp(modelo, [0, 39], y0, args=(p,), method = 'Radau', t_eval = t)

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

## IgM
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

plt.grid()
plt.plot(sol.t, np.log10(sol.y[15] + 1), color = 'pink')
plt.title("Células Natural Killers")
plt.xlabel("t (dias)")
plt.ylabel("$NK$")

x = nk[nk.type == 'mean']['x']
y = np.log10(nk[nk.type == 'mean']['y'] + 1)

nk_up = np.log10(nk[nk.type == 'up']['y'] + 1)
nk_down = np.log10(nk[nk.type == 'down']['y'] + 1)
nk_mean = np.log10(nk[nk.type == 'mean']['y'] + 1)

y_error = [nk_mean.to_numpy() - nk_down.to_numpy(), nk_up.to_numpy() - nk_mean.to_numpy()]

plt.errorbar(x,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='pink', capsize=4, elinewidth=1)

plt.show()

## Células TCD4+
dataset_tcd4 = pd.read_csv(r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\meus_dados\TCD4\TCD4_moderado.csv", sep = ',')

plt.grid()
plt.plot(sol.t, np.log10(sol.y[5] + 1), color = 'red')
plt.title("Células T Helpers")
plt.xlabel("t (dias)")
plt.ylabel("$TCD4+$")

x = dataset_tcd4[dataset_tcd4.type == 'mean']['x']
y = np.log10(dataset_tcd4[dataset_tcd4.type == 'mean']['y'] + 1)

dataset_tcd4_up = np.log10(dataset_tcd4[dataset_tcd4.type == 'up']['y'] + 1)
dataset_tcd4_down = np.log10(dataset_tcd4[dataset_tcd4.type == 'down']['y'] + 1)
dataset_tcd4_mean = np.log10(dataset_tcd4[dataset_tcd4.type == 'mean']['y'] + 1)

y_error = [dataset_tcd4_mean.to_numpy() - dataset_tcd4_down.to_numpy(), dataset_tcd4_up.to_numpy() - dataset_tcd4_mean.to_numpy()]

plt.errorbar(x,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='red', capsize=4, elinewidth=1)

plt.show()

## Células TCD8+
dataset_tcd8 = pd.read_csv(r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\meus_dados\TCD8\TCD8_moderado.csv", sep = ',')

plt.grid()
plt.plot(sol.t, np.log10(sol.y[7] + 1), color = 'orange')
plt.title("Células T Killers")
plt.xlabel("t (dias)")
plt.ylabel("$TCD8+$")

x = dataset_tcd8[dataset_tcd8.type == 'mean']['x']
y = np.log10(dataset_tcd8[dataset_tcd8.type == 'mean']['y'] + 1)

dataset_tcd8_up = np.log10(dataset_tcd8[dataset_tcd8.type == 'up']['y'] + 1)
dataset_tcd8_down = np.log10(dataset_tcd8[dataset_tcd8.type == 'down']['y'] + 1)
dataset_tcd8_mean = np.log10(dataset_tcd8[dataset_tcd8.type == 'mean']['y'] + 1)

y_error = [dataset_tcd8_mean.to_numpy() - dataset_tcd8_down.to_numpy(), dataset_tcd8_up.to_numpy() - dataset_tcd8_mean.to_numpy()]

plt.errorbar(x,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='orange', capsize=4, elinewidth=1)

plt.show()

## Células B
dataset_b = pd.read_csv(r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\meus_dados\Células B\cellB_moderado.csv", sep = ',')

plt.grid()
plt.plot(sol.t, np.log10(sol.y[8] + 1), color = 'green')
plt.title("Células B")
plt.xlabel("t (dias)")
plt.ylabel("$B$")

x = dataset_b[dataset_b.type == 'mean']['x']
y = np.log10(dataset_b[dataset_b.type == 'mean']['y'] + 1)

dataset_B_up = np.log10(dataset_b[dataset_b.type == 'up']['y'] + 1)
dataset_B_down = np.log10(dataset_b[dataset_b.type == 'down']['y'] + 1)
dataset_B_mean = np.log10(dataset_b[dataset_b.type == 'mean']['y'] + 1)

y_error = [dataset_B_mean.to_numpy() - dataset_B_down.to_numpy(), dataset_B_up.to_numpy() - dataset_B_mean.to_numpy()]

plt.errorbar(x,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='green', capsize=4, elinewidth=1)

plt.show()