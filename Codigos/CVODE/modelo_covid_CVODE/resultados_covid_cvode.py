import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

# Dados Experimentais
dataset_nk = pd.read_csv(r'/mnt/c/Users/mique/OneDrive/Documentos/UFJF/Modelagem Fisiologica/Data/covid_moderado/NK_covid_moderado.csv', sep = ',')
dataset_viremia = pd.read_csv(r'/mnt/c/Users/mique/OneDrive/Documentos/UFJF/Modelagem Fisiologica/Data/viremia_covid.csv', sep = ',')
dataset_igm = pd.read_csv(r'/mnt/c/Users/mique/OneDrive/Documentos/UFJF/Modelagem Fisiologica/Data/covid_moderado/IgM_covid_moderado.csv', sep = ',')
dataset_igg = pd.read_csv(r'/mnt/c/Users/mique/OneDrive/Documentos/UFJF/Modelagem Fisiologica/Data/covid_moderado/IgG_covid_moderado.csv', sep = ',')
dataset_il6 = pd.read_csv(r'/mnt/c/Users/mique/OneDrive/Documentos/UFJF/Modelagem Fisiologica/Data/citocinas_covid.csv', sep = ',')
dataset_tcd4 = pd.read_csv(r'/mnt/c/Users/mique/OneDrive/Documentos/UFJF/Modelagem Fisiologica/Data/covid_moderado/TCD4_covid_moderado.csv', sep = ',')
dataset_tcd8 = pd.read_csv(r'/mnt/c/Users/mique/OneDrive/Documentos/UFJF/Modelagem Fisiologica/Data/covid_moderado/TCD8_covid_moderado.csv', sep = ',')
dataset_b = pd.read_csv(r'/mnt/c/Users/mique/OneDrive/Documentos/UFJF/Modelagem Fisiologica/Data/covid_moderado/Bcell_covid_moderado.csv', sep = ',')

dataset_nk['y'] = dataset_nk['y'] / (10**3)
dataset_tcd4['y'] = dataset_tcd4['y'] / (10**3)
dataset_tcd8['y'] = dataset_tcd8['y'] / (10**3)
dataset_b['y'] = dataset_b['y'] / (10**3)

# Lendo o .csv com os resultados
df = pd.read_csv(r"/mnt/c/Users/mique/OneDrive/Documentos/UFJF/Modelagem Fisiologica/Codigos/CVODE/modelo_covid_CVODE/Resultados_simulacoes/teste.csv")

# GRÁFICOS
fig, ax = plt.subplots(2, 4, figsize = (22, 10))

## Viremia
ax[0, 0].plot(df['Time'], np.log10(df['V'] + 1), color='red')
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

ax[0, 0].errorbar(x + 4.15, y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='red', capsize=4, elinewidth=1)


## Citocinas
ax[0, 1].plot(df['Time'], np.log10(df['C'] + 1), color='blue')
ax[0, 1].set_title("Citocinas")
ax[0, 1].set_xlabel("t (dias)")
ax[0, 1].set_ylabel("$log_{10} (pg/ml + 1)$")
ax[0, 1].grid()

x = dataset_il6[dataset_il6.type == 'mean']['x']
y = np.log10(dataset_il6[dataset_il6.type == 'mean']['y'])

dataset_il6_up = np.log10(dataset_il6[dataset_il6.type == 'up']['y'])
dataset_il6_down = np.log10(dataset_il6[dataset_il6.type == 'down']['y'])
dataset_il6_mean = np.log10(dataset_il6[dataset_il6.type == 'mean']['y'])

y_error = [dataset_il6_mean.to_numpy() - dataset_il6_down.to_numpy(), dataset_il6_up.to_numpy() - dataset_il6_mean.to_numpy()]

ax[0, 1].errorbar(x + 4.15, y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='blue', capsize=4, elinewidth=1)


## IgG
ax[0, 2].plot(df['Time'], np.log10(df['IgG'] + 1), color='brown')
ax[0, 2].set_title("Anticorpos IgG")
ax[0, 2].set_xlabel("t (dias)")
ax[0, 2].set_ylabel("$log_{10} (AU/ml + 1)$")
ax[0, 2].grid()

x = dataset_igg[dataset_igg.type == 'mean']['x']
y = np.log10(dataset_igg[dataset_igg.type == 'mean']['y'] + 1)

dataset_igg_up = np.log10(dataset_igg[dataset_igg.type == 'up']['y'] + 1)
dataset_igg_down = np.log10(dataset_igg[dataset_igg.type == 'down']['y'] + 1)
dataset_igg_mean = np.log10(dataset_igg[dataset_igg.type == 'mean']['y'] + 1)

y_error = [dataset_igg_mean.to_numpy() - dataset_igg_down.to_numpy(), dataset_igg_up.to_numpy() - dataset_igg_mean.to_numpy()]

ax[0, 2].errorbar(x + 4.15, y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='brown', capsize=4, elinewidth=1)


## IgM
ax[0, 3].plot(df['Time'], np.log10(df['IgM'] + 1), color='orange')
ax[0, 3].set_title("Anticorpos IgM")
ax[0, 3].set_xlabel("t (dias)")
ax[0, 3].set_ylabel("$log_{10} (AU/ml + 1)$")
ax[0, 3].grid()

x = dataset_igm[dataset_igm.type == 'mean']['x']
y = np.log10(dataset_igm[dataset_igm.type == 'mean']['y'] + 1)

dataset_igm_up = np.log10(dataset_igm[dataset_igm.type == 'up']['y'] + 1) 
dataset_igm_down = np.log10(dataset_igm[dataset_igm.type == 'down']['y'] + 1)
dataset_igm_mean = np.log10(dataset_igm[dataset_igm.type == 'mean']['y'] + 1)

y_error = [dataset_igm_mean.to_numpy() - dataset_igm_down.to_numpy(), dataset_igm_up.to_numpy() - dataset_igm_mean.to_numpy()]

ax[0, 3].errorbar(x + 4.15, y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='orange', capsize=4, elinewidth=1)


## Natural Killers
ax[1, 0].plot(df['Time'], np.log10(df['NK'] + 1), color='pink')
ax[1, 0].set_title("Células Natural Killers")
ax[1, 0].set_xlabel("t (dias)")
ax[1, 0].set_ylabel("$log_{10} (copias/ml + 1)$")
ax[1, 0].grid()

x = dataset_nk[dataset_nk.type == 'mean']['x']
y = np.log10(dataset_nk[dataset_nk.type == 'mean']['y'] + 1)

dataset_nk_up = np.log10(dataset_nk[dataset_nk.type == 'up']['y'] + 1)
dataset_nk_down = np.log10(dataset_nk[dataset_nk.type == 'down']['y'] + 1)
dataset_nk_mean = np.log10(dataset_nk[dataset_nk.type == 'mean']['y'] + 1)

y_error = [dataset_nk_mean.to_numpy() - dataset_nk_down.to_numpy(), dataset_nk_up.to_numpy() - dataset_nk_mean.to_numpy()]

ax[1, 0].errorbar(x + 4.15,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='pink', capsize=4, elinewidth=1)


## Células TCD4+

ax[1, 1].plot(df['Time'], np.log10(df['ThE'] + 1), color='red')
ax[1, 1].set_title("Células T Helpers")
ax[1, 1].set_xlabel("t (dias)")
ax[1, 1].set_ylabel("$log_{10} (copias/ml + 1)$")
ax[1, 1].grid()

x = dataset_tcd4[dataset_tcd4.type == 'mean']['x']
y = np.log10(dataset_tcd4[dataset_tcd4.type == 'mean']['y'] + 1)

dataset_tcd4_up = np.log10(dataset_tcd4[dataset_tcd4.type == 'up']['y'] + 1)
dataset_tcd4_down = np.log10(dataset_tcd4[dataset_tcd4.type == 'down']['y'] + 1)
dataset_tcd4_mean = np.log10(dataset_tcd4[dataset_tcd4.type == 'mean']['y'] + 1)

y_error = [dataset_tcd4_mean.to_numpy() - dataset_tcd4_down.to_numpy(), dataset_tcd4_up.to_numpy() - dataset_tcd4_mean.to_numpy()]

ax[1, 1].errorbar(x + 4.15,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='red', capsize=4, elinewidth=1)


## Células TCD8+

ax[1, 2].grid()
ax[1, 2].plot(df['Time'], np.log10(df['TkE'] + 1), color='orange')
ax[1, 2].set_title("Células T Killers")
ax[1, 2].set_xlabel("t (dias)")
ax[1, 2].set_ylabel("$log_{10} (copias/ml + 1)$")

x = dataset_tcd8[dataset_tcd8.type == 'mean']['x']
y = np.log10(dataset_tcd8[dataset_tcd8.type == 'mean']['y'] + 1)

dataset_tcd8_up = np.log10(dataset_tcd8[dataset_tcd8.type == 'up']['y'] + 1)
dataset_tcd8_down = np.log10(dataset_tcd8[dataset_tcd8.type == 'down']['y'] + 1)
dataset_tcd8_mean = np.log10(dataset_tcd8[dataset_tcd8.type == 'mean']['y'] + 1)

y_error = [dataset_tcd8_mean.to_numpy() - dataset_tcd8_down.to_numpy(), dataset_tcd8_up.to_numpy() - dataset_tcd8_mean.to_numpy()]

ax[1, 2].errorbar(x + 4.15,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='orange', capsize=4, elinewidth=1)


## Células B

ax[1, 3].grid()
ax[1, 3].plot(df['Time'], np.log10(df['B'] + 1), color='green')
ax[1, 3].set_title("Células B")
ax[1, 3].set_xlabel("t (dias)")
ax[1, 3].set_ylabel("$log_{10} (copias/ml + 1)$")

x = dataset_b[dataset_b.type == 'mean']['x']
y = np.log10(dataset_b[dataset_b.type == 'mean']['y'] + 1)

dataset_B_up = np.log10(dataset_b[dataset_b.type == 'up']['y'] + 1)
dataset_B_down = np.log10(dataset_b[dataset_b.type == 'down']['y'] + 1)
dataset_B_mean = np.log10(dataset_b[dataset_b.type == 'mean']['y'] + 1)

y_error = [dataset_B_mean.to_numpy() - dataset_B_down.to_numpy(), dataset_B_up.to_numpy() - dataset_B_mean.to_numpy()]

ax[1, 3].errorbar(x + 4.15,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='green', capsize=4, elinewidth=1)

plt.savefig(r"/mnt/c/Users/mique/OneDrive/Documentos/UFJF/Modelagem Fisiologica/Codigos/CVODE/modelo_covid_CVODE/Resultados_simulacoes/Graficos/teste.png")