import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from covid19_modelo_moderado import modelo, y0, pars, params_ajs

# Carregando parâmetros ótimos
params_otimos = np.load(r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\params_otimos\parametros_otimos.npy")
 
p = pars.copy()
for i, key in enumerate(params_ajs):
    p[key] = params_otimos[i]
    

# Rodando o modelo com os parâmetros ótimos
t = np.linspace(0, 39, 3500)
sol = solve_ivp(modelo, [0, 39], y0, args=(p,), method = 'Radau', t_eval = t)

# GRÁFICOS
fig, ax = plt.subplots(2, 4, figsize = (22, 12))

## Viremia
ax[0, 0].plot(sol.t, np.log10(sol.y[0] + 1), color='red')
ax[0, 0].set_title("Vírus")
ax[0, 0].set_xlabel("t (dias)")
ax[0, 0].set_ylabel("V")
ax[0, 0].grid()

dataset_viremia = pd.read_csv(r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\covid19_model\data\Viral_load.csv", sep = ',')

x = dataset_viremia[dataset_viremia.type == 'mean']['x']
y = np.log10(dataset_viremia[dataset_viremia.type == 'mean']['y'] + 1)

dataset_viremia_up = np.log10(dataset_viremia[dataset_viremia.type == 'up']['y'] + 1)
dataset_viremia_down = np.log10(dataset_viremia[dataset_viremia.type == 'down']['y'] + 1)
dataset_viremia_mean = np.log10(dataset_viremia[dataset_viremia.type == 'mean']['y'] + 1)

y_error = [dataset_viremia_mean.to_numpy() - dataset_viremia_down.to_numpy(), dataset_viremia_up.to_numpy() - dataset_viremia_mean.to_numpy()]

ax[0, 0].errorbar(x,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='red', capsize=4, elinewidth=1)


## Citocinas
ax[0, 1].plot(sol.t, sol.y[14], color='blue')
ax[0, 1].set_title("Citocinas")
ax[0, 1].set_xlabel("t (dias)")
ax[0, 1].set_ylabel("C")
ax[0, 1].grid()

dataset_il6 = pd.read_csv(r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\covid19_model\data\dataset_il6_survivor.csv", sep = ',')

x = dataset_il6[dataset_il6.type == 'mean']['x']
y = dataset_il6[dataset_il6.type == 'mean']['y']

dataset_il6_up = dataset_il6[dataset_il6.type == 'up']['y']
dataset_il6_down = dataset_il6[dataset_il6.type == 'down']['y']
dataset_il6_mean = dataset_il6[dataset_il6.type == 'mean']['y']

y_error = [dataset_il6_mean.to_numpy() - dataset_il6_down.to_numpy(), dataset_il6_up.to_numpy() - dataset_il6_mean.to_numpy()]

ax[0, 1].errorbar(x,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='blue', capsize=4, elinewidth=1)


## IgG
ax[0, 2].plot(sol.t, np.log10(sol.y[13] + 1), color='brown')
ax[0, 2].set_title("Anticorpos IgG")
ax[0, 2].set_xlabel("t (dias)")
ax[0, 2].set_ylabel("$I_{gG}$")
ax[0, 2].grid()

dataset_IgG = pd.read_csv(r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\meus_dados\IgG\IgG_moderado.csv", sep = ',')

x = dataset_IgG[dataset_IgG.type == 'mean']['x']
y = np.log10(dataset_IgG[dataset_IgG.type == 'mean']['y'] + 1)

dataset_IgG_up = np.log10(dataset_IgG[dataset_IgG.type == 'up']['y'] + 1)
dataset_IgG_down = np.log10(dataset_IgG[dataset_IgG.type == 'down']['y'] + 1)
dataset_IgG_mean = np.log10(dataset_IgG[dataset_IgG.type == 'mean']['y'] + 1)

y_error = [dataset_IgG_mean.to_numpy() - dataset_IgG_down.to_numpy(), dataset_IgG_up.to_numpy() - dataset_IgG_mean.to_numpy()]

ax[0, 2].errorbar(x,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='brown', capsize=4, elinewidth=1)


## IgM
ax[0, 3].plot(sol.t, np.log10(sol.y[12] + 1), color='orange')
ax[0, 3].set_title("Anticorpos IgM")
ax[0, 3].set_xlabel("t (dias)")
ax[0, 3].set_ylabel("$I_{gM}$")
ax[0, 3].grid()

dataset_IgM = pd.read_csv(r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\meus_dados\IgM\IgM_moderado.csv", sep = ',')

x = dataset_IgM[dataset_IgM.type == 'mean']['x']
y = np.log10(dataset_IgM[dataset_IgM.type == 'mean']['y'] + 1)

dataset_IgM_up = np.log10(dataset_IgM[dataset_IgM.type == 'up']['y'] + 1)
dataset_IgM_down = np.log10(dataset_IgM[dataset_IgM.type == 'down']['y'] + 1)
dataset_IgM_mean = np.log10(dataset_IgM[dataset_IgM.type == 'mean']['y'] + 1)

y_error = [dataset_IgM_mean.to_numpy() - dataset_IgM_down.to_numpy(), dataset_IgM_up.to_numpy() - dataset_IgM_mean.to_numpy()]

ax[0, 3].errorbar(x,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='orange', capsize=4, elinewidth=1)


## Natural Killers
dataset_NK = pd.read_csv(r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\meus_dados\NK\NK_moderado.csv", sep = ',')

ax[1, 0].grid()
ax[1, 0].plot(sol.t, np.log10(sol.y[15] + 1), color = 'pink')
ax[1, 0].set_title("Células Natural Killers")
ax[1, 0].set_xlabel("t (dias)")
ax[1, 0].set_ylabel("$NK$")

x = dataset_NK[dataset_NK.type == 'mean']['x']
y = np.log10(dataset_NK[dataset_NK.type == 'mean']['y'] + 1)

dataset_NK_up = np.log10(dataset_NK[dataset_NK.type == 'up']['y'] + 1)
dataset_NK_down = np.log10(dataset_NK[dataset_NK.type == 'down']['y'] + 1)
dataset_NK_mean = np.log10(dataset_NK[dataset_NK.type == 'mean']['y'] + 1)

y_error = [dataset_NK_mean.to_numpy() - dataset_NK_down.to_numpy(), dataset_NK_up.to_numpy() - dataset_NK_mean.to_numpy()]

ax[1, 0].errorbar(x,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='pink', capsize=4, elinewidth=1)


## Células TCD4+
dataset_tcd4 = pd.read_csv(r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\meus_dados\TCD4\TCD4_moderado.csv", sep = ',')

ax[1, 1].grid()
ax[1, 1].plot(sol.t, np.log10(sol.y[5] + 1), color = 'red')
ax[1, 1].set_title("Células T Helpers")
ax[1, 1].set_xlabel("t (dias)")
ax[1, 1].set_ylabel("$TCD4+$")

x = dataset_tcd4[dataset_tcd4.type == 'mean']['x']
y = np.log10(dataset_tcd4[dataset_tcd4.type == 'mean']['y'] + 1)

dataset_tcd4_up = np.log10(dataset_tcd4[dataset_tcd4.type == 'up']['y'] + 1)
dataset_tcd4_down = np.log10(dataset_tcd4[dataset_tcd4.type == 'down']['y'] + 1)
dataset_tcd4_mean = np.log10(dataset_tcd4[dataset_tcd4.type == 'mean']['y'] + 1)

y_error = [dataset_tcd4_mean.to_numpy() - dataset_tcd4_down.to_numpy(), dataset_tcd4_up.to_numpy() - dataset_tcd4_mean.to_numpy()]

ax[1, 1].errorbar(x,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='red', capsize=4, elinewidth=1)


## Células TCD8+
dataset_tcd8 = pd.read_csv(r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\meus_dados\TCD8\TCD8_moderado.csv", sep = ',')

ax[1, 2].grid()
ax[1, 2].plot(sol.t, np.log10(sol.y[7] + 1), color = 'orange')
ax[1, 2].set_title("Células T Killers")
ax[1, 2].set_xlabel("t (dias)")
ax[1, 2].set_ylabel("$TCD8+$")

x = dataset_tcd8[dataset_tcd8.type == 'mean']['x']
y = np.log10(dataset_tcd8[dataset_tcd8.type == 'mean']['y'] + 1)

dataset_tcd8_up = np.log10(dataset_tcd8[dataset_tcd8.type == 'up']['y'] + 1)
dataset_tcd8_down = np.log10(dataset_tcd8[dataset_tcd8.type == 'down']['y'] + 1)
dataset_tcd8_mean = np.log10(dataset_tcd8[dataset_tcd8.type == 'mean']['y'] + 1)

y_error = [dataset_tcd8_mean.to_numpy() - dataset_tcd8_down.to_numpy(), dataset_tcd8_up.to_numpy() - dataset_tcd8_mean.to_numpy()]

ax[1, 2].errorbar(x,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='orange', capsize=4, elinewidth=1)


## Células B
dataset_b = pd.read_csv(r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\meus_dados\Células B\cellB_moderado.csv", sep = ',')

ax[1, 3].grid()
ax[1, 3].plot(sol.t, np.log10(sol.y[8] + 1), color = 'green')
ax[1, 3].set_title("Células B")
ax[1, 3].set_xlabel("t (dias)")
ax[1, 3].set_ylabel("$B$")

x = dataset_b[dataset_b.type == 'mean']['x']
y = np.log10(dataset_b[dataset_b.type == 'mean']['y'] + 1)

dataset_B_up = np.log10(dataset_b[dataset_b.type == 'up']['y'] + 1)
dataset_B_down = np.log10(dataset_b[dataset_b.type == 'down']['y'] + 1)
dataset_B_mean = np.log10(dataset_b[dataset_b.type == 'mean']['y'] + 1)

y_error = [dataset_B_mean.to_numpy() - dataset_B_down.to_numpy(), dataset_B_up.to_numpy() - dataset_B_mean.to_numpy()]

ax[1, 3].errorbar(x,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='green', capsize=4, elinewidth=1)

plt.show()