import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ler os arquivos CSV
df = pd.read_csv('model_results.csv')
data = {}
data['cd4'] = pd.read_csv("https://raw.githubusercontent.com/ruyfreis/covid19_model/main/data/CD4_Critical.csv")
data['cd8'] = pd.read_csv("https://raw.githubusercontent.com/ruyfreis/covid19_model/main/data/CD8_Critical.csv")
data['virus'] = pd.read_csv("https://raw.githubusercontent.com/ruyfreis/covid19_model/main/data/Viral_load.csv")
data['igg'] = pd.read_csv("https://raw.githubusercontent.com/ruyfreis/covid19_model/main/data/IgG_data.csv")
data['igm'] = pd.read_csv("https://raw.githubusercontent.com/ruyfreis/covid19_model/main/data/IgM_data.csv")

CD4 = data['cd4']
CD8 = data['cd8']
virus = data['virus']
IgG = data['igg']
IgM = data['igm']

# Plotar os dados
plt.figure(figsize=(10, 6))
plt.plot(df['Time'], df['The'])
plt.title('CD4')
plt.xlabel('t')
plt.ylabel('The')
plt.grid(True)

x = CD4[CD4.type == 'mean']['x'] + 5
y = CD4[CD4.type == 'mean']['y']

CD4_up = CD4[CD4.type == 'up']['y']
CD4_down = CD4[CD4.type == 'down']['y']
CD4_mean = CD4[CD4.type == 'mean']['y']

y_error = [CD4_mean.to_numpy() - CD4_down.to_numpy(), CD4_up.to_numpy() - CD4_mean.to_numpy()]

plt.errorbar(x,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='orange', capsize=4, elinewidth=1)

plt.savefig('cd4.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df['Time'], df['Tke'])
plt.title('CD8')
plt.xlabel('t')
plt.ylabel('Tke')
plt.grid(True)

x = CD8[CD8.type == 'mean']['x'] + 5
y = CD8[CD8.type == 'mean']['y']

CD8_up = CD8[CD8.type == 'up']['y']
CD8_down = CD8[CD8.type == 'down']['y']
CD8_mean = CD8[CD8.type == 'mean']['y']

y_error = [CD8_mean.to_numpy() - CD8_down.to_numpy(), CD8_up.to_numpy() - CD8_mean.to_numpy()]

plt.errorbar(x,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='orange', capsize=4, elinewidth=1)

plt.savefig('cd8.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df['Time'], np.log10(df['V'] + 1))
plt.title('Virus')
plt.xlabel('t')
plt.ylabel('V')
plt.grid(True)

x = virus[virus.type == 'mean']['x'] + 5
y = np.log10(virus[virus.type == 'mean']['y'] + 1)

virus_up = np.log10(virus[virus.type == 'up']['y'] + 1)
virus_down = np.log10(virus[virus.type == 'down']['y'] + 1)
virus_mean = np.log10(virus[virus.type == 'mean']['y'] + 1)

y_error = [virus_mean.to_numpy() - virus_down.to_numpy(), virus_up.to_numpy() - virus_mean.to_numpy()]

plt.errorbar(x,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='orange', capsize=4, elinewidth=1)

plt.savefig('virus.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df['Time'], np.log2(df['A'] + 1))
plt.title('Anticorpos')
plt.xlabel('t')
plt.ylabel('A')
plt.grid(True)

x = IgG[IgG.type == 'mean']['x'] + 5
y = np.log2(IgG[IgG.type == 'mean']['y'] + IgM[IgM.type == 'mean']['y'] + 1)

ant_up = np.log2(IgG[IgG.type == 'up']['y'] + IgM[IgM.type == 'up']['y'] + 1)
ant_down = np.log2(IgG[IgG.type == 'down']['y'] + IgM[IgM.type == 'down']['y'] + 1)
ant_mean = np.log2(IgG[IgG.type == 'mean']['y'] + IgM[IgM.type == 'mean']['y'] + 1)

y_error = [ant_mean.to_numpy() - ant_down.to_numpy(), ant_up.to_numpy() - ant_mean.to_numpy()]

plt.errorbar(x,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='orange', capsize=4, elinewidth=1)

plt.savefig('ant.png', bbox_inches='tight')
plt.show()
