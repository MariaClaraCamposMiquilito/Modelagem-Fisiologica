import json
import numpy as np
from scipy.integrate import solve_ivp
from modelo_covid19_ajustado import modelo, carrega_dados, pars, y0
import matplotlib.pyplot as plt

# Carregando os parâmetros ajustados
with open(r'C:\Users\karla\OneDrive\Documentos\UFJF\Modelagem Fisiologica\parametros_otimos,json', 'r') as f:
    pars_otimo = json.load(f)

pars_otimo['Ap0'] = 1.0e6
pars_otimo['ThN0'] = 1.0e6
pars_otimo['TkN0'] = 5.0e5
pars_otimo['B0'] = 2.5e5
pars_otimo['Nmax'] = 5.0e5 

t_eval = np.linspace(0, 35, 3500)
sol = solve_ivp(modelo, [0, 35], y0, args=(pars_otimo,), method='Radau', t_eval=t_eval)

viremia, il6, igg, igm = carrega_dados()


# Viremia
plt.figure(figsize = (8, 5))
plt.plot(sol.t, np.log10(sol.y[0] + 1), color='red', label='Modelo (V)')
df_v = viremia[viremia.type == 'mean']
x = df_v['x']
y = np.log10(df_v['y'])

y_up = np.log10(viremia[viremia.type == 'up']['y'] + 1)
y_down = np.log10(viremia[viremia.type == 'down']['y'] + 1)
y_err = [y - y_down.values, y_up.values - y]

plt.errorbar(x, y, yerr=y_err, fmt='o', color='red', capsize=4, label='Dados')
plt.title("Vírus"); plt.xlabel("t (dias)"); plt.ylabel("$\log_{10}(V+1)$"); plt.grid(True); plt.legend()
plt.show()

# Citocinas
plt.figure(figsize = (8, 5))
plt.plot(sol.t, np.log10(sol.y[14] + 1), color='blue', label='Modelo (C)')
df_c = il6[il6.type == 'mean']
x = df_c['x']
y = np.log10(df_c['y'])

y_up = np.log10(il6[il6.type == 'up']['y'] + 1)
y_down = np.log10(il6[il6.type == 'down']['y'] + 1)
y_err = [y - y_down.values, y_up.values - y]

plt.errorbar(x, y, yerr=y_err, fmt='o', color='blue', capsize=4, label='Dados')
plt.title("Citocinas"); plt.xlabel("t (dias)"); plt.ylabel("C"); plt.grid(True); plt.legend()
plt.show()

# IgG
plt.figure(figsize=(8, 5))
plt.plot(sol.t, np.log2(sol.y[13] + 1), color='brown', label='Modelo ($I_{gG}$)')

df_g = igg[igg.type == 'mean']
x = df_g['x']
y = np.log2(df_g['y'] + 1)

y_up = np.log2(igg[igg.type == 'up']['y'] + 1)
y_down = np.log2(igg[igg.type == 'down']['y'] + 1)
y_err = [y - y_down.values, y_up.values - y]

plt.errorbar(x, y, yerr=y_err, fmt='o', color='brown', capsize=4, label='Dados')
plt.title("Anticorpos IgG"); plt.xlabel("t (dias)"); plt.ylabel("$\log_{2}(I_{gG}+1)$"); plt.grid(True); plt.legend()
plt.show()

# IgM
plt.figure(figsize=(8, 5))
plt.plot(sol.t, np.log2(sol.y[12] + 1), color='orange', label='Modelo ($I_{gM}$)')

# Dados Experimentais
df_m = igm[igm.type == 'mean']
x = df_m['x']
y = np.log2(df_m['y'] + 1)

y_up = np.log2(igm[igm.type == 'up']['y'] + 1)
y_down = np.log2(igm[igm.type == 'down']['y'] + 1)
y_err = [y - y_down.values, y_up.values - y]

plt.errorbar(x, y, yerr=y_err, fmt='o', color='orange', capsize=4, label='Dados')
plt.title("Anticorpos IgM"); plt.xlabel("t (dias)"); plt.ylabel("$\log_{2}(I_{gM}+1)$"); plt.grid(True); plt.legend()
plt.show()

# Células NK
plt.figure(figsize=(8, 5))
plt.plot(sol.t, sol.y[15], color='green', label='Modelo (NK)')

plt.title("Células Natural Killer (NK)")
plt.xlabel("t (dias)")
plt.ylabel("Células/mL")
plt.grid(True)
plt.legend()
plt.show()