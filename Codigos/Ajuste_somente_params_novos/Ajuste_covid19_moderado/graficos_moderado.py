import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from covid19_modelo_moderado import modelo, y0, pars, params_ajs, carrega_dados

dataset_nk, dataset_viremia, dataset_igm, dataset_igg, dataset_il6, dataset_tcd4, dataset_tcd8, dataset_b = carrega_dados()

# Carregando parâmetros ótimos
params_otimos = np.load(r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\params_otimos\parametros_otimos.npy")

#params_otimos[3] = 6e
#params_otimos[3] = 1e5 
#params_otimos[1] = 0.9
#params_otimos[4] = 0.07
#params_otimos[17] = 4e5

p = pars.copy()
for i, key in enumerate(params_ajs):
    p[key] = params_otimos[i]

# Rodando o modelo com os parâmetros ótimos
t = np.linspace(0, 40, 3500)

y0_local = y0.copy()
y0_local[0]  = p['V0']
y0_local[1]  = p['Ap0']
y0_local[8]  = p['B0']
y0_local[15] = p['NK0']

sol = solve_ivp(modelo, [0, 40], y0_local, args=(p,), method='Radau', t_eval=t)

# GRÁFICOS
fig, ax = plt.subplots(2, 4, figsize = (22, 12))

## Viremia
ax[0, 0].plot(sol.t + 4.15, np.log10(sol.y[0] + 1), color='red')
ax[0, 0].set_title("Viremia")
ax[0, 0].set_xlabel("t (dias)")
ax[0, 0].set_ylabel("$log_{10} (cópias/ml + 1)$")
ax[0, 0].grid()


x = dataset_viremia[dataset_viremia.type == 'mean']['x']
y = np.log10(dataset_viremia[dataset_viremia.type == 'mean']['y'] + 1)

dataset_viremia_up = np.log10(dataset_viremia[dataset_viremia.type == 'up']['y'] + 1)
dataset_viremia_down = np.log10(dataset_viremia[dataset_viremia.type == 'down']['y'] + 1)
dataset_viremia_mean = np.log10(dataset_viremia[dataset_viremia.type == 'mean']['y'] + 1)

y_error = [dataset_viremia_mean.to_numpy() - dataset_viremia_down.to_numpy(), dataset_viremia_up.to_numpy() - dataset_viremia_mean.to_numpy()]

ax[0, 0].errorbar(x + 4.15,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='red', capsize=4, elinewidth=1)


## Citocinas
ax[0, 1].plot(sol.t + 4.15, sol.y[14], color='blue')
ax[0, 1].set_title("Citocinas")
ax[0, 1].set_xlabel("t (dias)")
ax[0, 1].set_ylabel("$log_{10} (pg/ml)$")
ax[0, 1].grid()


x = dataset_il6[dataset_il6.type == 'mean']['x']
y = dataset_il6[dataset_il6.type == 'mean']['y']

dataset_il6_up = dataset_il6[dataset_il6.type == 'up']['y']
dataset_il6_down = dataset_il6[dataset_il6.type == 'down']['y']
dataset_il6_mean = dataset_il6[dataset_il6.type == 'mean']['y']

y_error = [dataset_il6_mean.to_numpy() - dataset_il6_down.to_numpy(), dataset_il6_up.to_numpy() - dataset_il6_mean.to_numpy()]

ax[0, 1].errorbar(x + 4.15,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='blue', capsize=4, elinewidth=1)


## IgG
ax[0, 2].plot(sol.t + 4.15, np.log10(sol.y[13] + 1), color='brown')
ax[0, 2].set_title("Anticorpos IgG")
ax[0, 2].set_xlabel("t (dias)")
ax[0, 2].set_ylabel("$log_{10} (AU/ml)$")
ax[0, 2].grid()


x = dataset_igg[dataset_igg.type == 'mean']['x']
y = np.log10(dataset_igg[dataset_igg.type == 'mean']['y'] + 1)

dataset_igg_up = np.log10(dataset_igg[dataset_igg.type == 'up']['y'] + 1)
dataset_igg_down = np.log10(dataset_igg[dataset_igg.type == 'down']['y'] + 1)
dataset_igg_mean = np.log10(dataset_igg[dataset_igg.type == 'mean']['y'] + 1)

y_error = [dataset_igg_mean.to_numpy() - dataset_igg_down.to_numpy(), dataset_igg_up.to_numpy() - dataset_igg_mean.to_numpy()]

ax[0, 2].errorbar(x + 4.15,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='brown', capsize=4, elinewidth=1)


## IgM
ax[0, 3].plot(sol.t + 4.15, np.log10(sol.y[12] + 1), color='orange')
ax[0, 3].set_title("Anticorpos IgM")
ax[0, 3].set_xlabel("t (dias)")
ax[0, 3].set_ylabel("$log_{10} (AU/ml)$")
ax[0, 3].grid()


x = dataset_igm[dataset_igm.type == 'mean']['x']
y = np.log10(dataset_igm[dataset_igm.type == 'mean']['y'] + 1)

dataset_igm_up = np.log10(dataset_igm[dataset_igm.type == 'up']['y'] + 1) 
dataset_igm_down = np.log10(dataset_igm[dataset_igm.type == 'down']['y'] + 1)
dataset_igm_mean = np.log10(dataset_igm[dataset_igm.type == 'mean']['y'] + 1)

y_error = [dataset_igm_mean.to_numpy() - dataset_igm_down.to_numpy(), dataset_igm_up.to_numpy() - dataset_igm_mean.to_numpy()]

ax[0, 3].errorbar(x + 4.15,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='orange', capsize=4, elinewidth=1)


## Natural Killers

ax[1, 0].grid()
ax[1, 0].plot(sol.t + 4.15, np.log10(sol.y[15] + 1), color = 'pink')
ax[1, 0].set_title("Células Natural Killers")
ax[1, 0].set_xlabel("t (dias)")
ax[1, 0].set_ylabel("$log_{10} (10³/ml)$")

x = dataset_nk[dataset_nk.type == 'mean']['x']
y = np.log10(dataset_nk[dataset_nk.type == 'mean']['y'] + 1)

dataset_nk_up = np.log10(dataset_nk[dataset_nk.type == 'up']['y'] + 1)
dataset_nk_down = np.log10(dataset_nk[dataset_nk.type == 'down']['y'] + 1)
dataset_nk_mean = np.log10(dataset_nk[dataset_nk.type == 'mean']['y'] + 1)

y_error = [dataset_nk_mean.to_numpy() - dataset_nk_down.to_numpy(), dataset_nk_up.to_numpy() - dataset_nk_mean.to_numpy()]

ax[1, 0].errorbar(x + 4.15,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='pink', capsize=4, elinewidth=1)


## Células TCD4+

ax[1, 1].grid()
ax[1, 1].plot(sol.t + 4.15, np.log10(sol.y[5] + 1), color = 'red')
ax[1, 1].set_title("Células T Helpers")
ax[1, 1].set_xlabel("t (dias)")
ax[1, 1].set_ylabel("$log_{10} (10³/ml)$")

x = dataset_tcd4[dataset_tcd4.type == 'mean']['x']
y = np.log10(dataset_tcd4[dataset_tcd4.type == 'mean']['y'] + 1)

dataset_tcd4_up = np.log10(dataset_tcd4[dataset_tcd4.type == 'up']['y'] + 1)
dataset_tcd4_down = np.log10(dataset_tcd4[dataset_tcd4.type == 'down']['y'] + 1)
dataset_tcd4_mean = np.log10(dataset_tcd4[dataset_tcd4.type == 'mean']['y'] + 1)

y_error = [dataset_tcd4_mean.to_numpy() - dataset_tcd4_down.to_numpy(), dataset_tcd4_up.to_numpy() - dataset_tcd4_mean.to_numpy()]

ax[1, 1].errorbar(x + 4.15,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='red', capsize=4, elinewidth=1)


## Células TCD8+

ax[1, 2].grid()
ax[1, 2].plot(sol.t + 4.15, np.log10(sol.y[7] + 1), color = 'orange')
ax[1, 2].set_title("Células T Killers")
ax[1, 2].set_xlabel("t (dias)")
ax[1, 2].set_ylabel("$log_{10} (10³/ml)$")

x = dataset_tcd8[dataset_tcd8.type == 'mean']['x']
y = np.log10(dataset_tcd8[dataset_tcd8.type == 'mean']['y'] + 1)

dataset_tcd8_up = np.log10(dataset_tcd8[dataset_tcd8.type == 'up']['y'] + 1)
dataset_tcd8_down = np.log10(dataset_tcd8[dataset_tcd8.type == 'down']['y'] + 1)
dataset_tcd8_mean = np.log10(dataset_tcd8[dataset_tcd8.type == 'mean']['y'] + 1)

y_error = [dataset_tcd8_mean.to_numpy() - dataset_tcd8_down.to_numpy(), dataset_tcd8_up.to_numpy() - dataset_tcd8_mean.to_numpy()]

ax[1, 2].errorbar(x + 4.15,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='orange', capsize=4, elinewidth=1)


## Células B

ax[1, 3].grid()
ax[1, 3].plot(sol.t + 4.15, np.log10(sol.y[8] + sol.y[11] + 1), color = 'green')
ax[1, 3].set_title("Células B")
ax[1, 3].set_xlabel("t (dias)")
ax[1, 3].set_ylabel("$log_{10} (10³/ml)$")

x = dataset_b[dataset_b.type == 'mean']['x']
y = np.log10(dataset_b[dataset_b.type == 'mean']['y'] + 1)

dataset_B_up = np.log10(dataset_b[dataset_b.type == 'up']['y'] + 1)
dataset_B_down = np.log10(dataset_b[dataset_b.type == 'down']['y'] + 1)
dataset_B_mean = np.log10(dataset_b[dataset_b.type == 'mean']['y'] + 1)

y_error = [dataset_B_mean.to_numpy() - dataset_B_down.to_numpy(), dataset_B_up.to_numpy() - dataset_B_mean.to_numpy()]

ax[1, 3].errorbar(x + 4.15,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='green', capsize=4, elinewidth=1)

plt.show()


plt.plot(sol.t, np.log10(sol.y[2]), label = "APCs")
plt.show()