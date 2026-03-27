import numpy as np
from scipy.integrate import solve_ivp
from modelo_covid19_ajustado import modelo, carrega_dados, pars, y0
import matplotlib.pyplot as plt

array_otimo = np.load(r'C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\params_otimos\parametros_otimos.npy')

pars_plot = pars.copy()
all_pars = [k for k in pars.keys() if k not in ['Ap0', 'ThN0', 'TkN0', 'B0', 'NK0']]

for i, key in enumerate(all_pars):
    pars_plot[key] = array_otimo[i]

t = np.linspace(0, 37, 3500)
sol = solve_ivp(modelo, [0, 37], y0, args=(pars_plot,), method = 'Radau', t_eval = t)

viremia, il6, igg, igm, nk = carrega_dados()

fig, ax = plt.subplots(4, 2, figsize = (10, 8))

# VIREMIA
ax[0, 0].plot(sol.t, np.log10(sol.y[0]), color = 'red', label = 'Modelo (V)')

df_v = viremia[viremia.type == 'mean']
x_v = df_v['x']
y_v = np.log10(df_v['y'] + 1)
y_up_v = np.log10(viremia[viremia.type == 'up']['y'] + 1)
y_down_v = np.log10(viremia[viremia.type == 'down']['y'] + 1)
y_err_v = [np.abs(y_v - y_down_v.values), np.abs(y_up_v.values - y_v)]

ax[0, 0].errorbar(x_v, y_v, yerr=y_err_v, fmt='o', color='red', capsize=4, label='Dados')
ax[0, 0].set_title('Vírus')
ax[0, 0].set_xlabel('t (dias)')
ax[0, 0].set_ylabel(r"$\log_{10}(V)$") 
ax[0, 0].grid(True)
ax[0, 0].legend()

# CITOCINAS
ax[0, 1].plot(sol.t, np.log10(sol.y[14]), color = 'blue', label = 'Modelo (C)')

df_c = il6[il6.type == 'mean']
x_c = df_c['x']
y_c = np.log10(df_c['y'] + 1)
y_up_c = np.log10(il6[il6.type == 'up']['y'] + 1)
y_down_c = np.log10(il6[il6.type == 'down']['y'] + 1)
y_err_c = [np.abs(y_c - y_down_c.values), np.abs(y_up_c.values - y_c)]

ax[0, 1].errorbar(x_c, y_c, yerr = y_err_c, fmt='o', color='blue', capsize=4, label='Dados')
ax[0, 1].set_title('Citocinas')
ax[0, 1].set_xlabel('t (dias)')
ax[0, 1].set_ylabel(r"$\log_{10}(C)$") 
ax[0, 1].grid(True)
ax[0, 1].legend()

# IgG
ax[1, 0].plot(sol.t, np.log10(sol.y[13]), color='brown', label='Modelo ($I_{gG}$)')
df_g = igg[igg.type == 'mean']
x_g = df_g['x']
y_g = np.log10(df_g['y'] + 1)
y_up_g = np.log10(igg[igg.type == 'up']['y'] + 1)
y_down_g = np.log10(igg[igg.type == 'down']['y'] + 1)
y_err_g = [np.abs(y_g - y_down_g.values), np.abs(y_up_g.values - y_g)]
ax[1, 0].errorbar(x_g, y_g, yerr=y_err_g, fmt='o', color='brown', capsize=4, label='Dados')
ax[1, 0].set_title('Anticorpos IgG')
ax[1, 0].set_xlabel('t (dias)')
ax[1, 0].set_ylabel(r'$\log_{10}(I_{gG})$')
ax[1, 0].grid(True)
ax[1, 0].legend()

# IgM
ax[1, 1].plot(sol.t, np.log10(sol.y[12]), color='orange', label='Modelo ($I_{gM}$)')
df_m = igm[igm.type == 'mean']
x_m = df_m['x']
y_m = np.log10(df_m['y'] + 1)
y_up_m = np.log10(igm[igm.type == 'up']['y'] + 1)
y_down_m = np.log10(igm[igm.type == 'down']['y'] + 1)
y_err_m = [np.abs(y_m - y_down_m.values), np.abs(y_up_m.values - y_m)]
ax[1, 1].errorbar(x_m, y_m, yerr=y_err_m, fmt='o', color='orange', capsize=4, label='Dados')
ax[1, 1].set_title('Anticorpos IgM')
ax[1, 1].set_xlabel('t (dias)')
ax[1, 1].set_ylabel(r'$\log_{10}(I_{gM})$')
ax[1, 1].grid(True)
ax[1, 1].legend()

# Células NK
ax[2, 0].plot(sol.t, np.log10(sol.y[15]), color='green', label='Modelo (NK)')
ax[2, 0].scatter(nk['x'], np.log10(nk['y']), label = 'Dados', color = 'green')
ax[2, 0].set_title('Células Natural Killer (NK)')
ax[2, 0].set_xlabel('t (dias)')
ax[2, 0].set_ylabel(r"$\log_{10}(NK)$") 
ax[2, 0].grid(True)
ax[2, 0].legend()

# Células infectadas
ax[2, 1].plot(sol.t, np.log10(sol.y[3]), color = 'pink', label = 'Células Infectadas')
ax[2, 1].set_xlabel('t (dias)')
ax[2, 1].set_ylabel("$\log_{10}(I)$")
ax[2, 1].set_title('Células Infectadas')
ax[2, 1].grid(True)
ax[2, 1].legend()
plt.tight_layout()

# Células T Killers
ax[3, 0].plot(sol.t, np.log10(sol.y[7]), color = 'purple', label = 'Células T Killers')
ax[3, 0].set_xlabel('t (dias)')
ax[3, 0].set_ylabel("$\log_{10}(T_{kE})$")
ax[3, 0].set_title('Células T Killers')
ax[3, 0].grid(True)
ax[3, 0].legend()
plt.tight_layout()


# Células Apresentadoras de Antígenos
ax[3, 1].plot(sol.t, np.log10(sol.y[2]), color = 'yellow', label = 'APCs')
ax[3, 1].set_xlabel('t (dias)')
ax[3, 1].set_ylabel("$\log_{10}(A_{pM})$")
ax[3, 1].set_title('Células Apresentadoras de Antígenos (APCs)')
ax[3, 1].grid(True)
ax[3, 1].legend()
plt.tight_layout()

plt.show()