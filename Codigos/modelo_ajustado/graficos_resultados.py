import numpy as np
from scipy.integrate import solve_ivp
# Importamos params_ajs para saber quais chaves foram ajustadas
from modelo_covid19_ajustado import modelo, carrega_dados, pars, y0, params_ajs
import matplotlib.pyplot as plt
import pandas as pd

# 1. Carregar os resultados
path = r'C:\Users\karla\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\modelo_ajustado'
array_otimo = np.load(path + r'\parametros_otimos.npy')

# 2. Reconstruir o dicionário de parâmetros e o y0
pars_plot = pars.copy()
keys_pars = [k for k in params_ajs] # Garante a mesma ordem do ajuste

for i, key in enumerate(keys_pars):
    pars_plot[key] = array_otimo[i]

y0_simulacao = [
    pars_plot.get('V0', y0[0]), pars_plot.get('Ap0', y0[1]), pars_plot.get('ApM0', y0[2]), 
    pars_plot.get('I0', y0[3]), pars_plot.get('ThN0', y0[4]), pars_plot.get('ThE0', y0[5]), 
    pars_plot.get('TkN0', y0[6]), pars_plot.get('TkE0', y0[7]), pars_plot.get('B0', y0[8]), 
    pars_plot.get('Ps0', y0[9]), pars_plot.get('Pl0', y0[10]), pars_plot.get('Bm0', y0[11]), 
    pars_plot.get('IgM0', y0[12]), pars_plot.get('IgG0', y0[13]), pars_plot.get('C0', y0[14]), 
    pars_plot.get('NK0', y0[15])
]

# 3. Simular com os parâmetros ótimos
t_eval = np.linspace(0, 37, 3500)
sol = solve_ivp(modelo, [0, 37], y0_simulacao, args=(pars_plot,), method='Radau', t_eval=t_eval)

# 4. Plotagem (Mantendo sua estrutura original)
viremia, il6, igg, igm, nk = carrega_dados()
fig, ax = plt.subplots(3, 2, figsize=(10, 8))

# Função auxiliar para evitar repetição de código no log e erro
def plot_data(ax_obj, data_df, sol_y, color, label, title, ylabel, is_nk=False):
    ax_obj.plot(sol.t, np.log10(sol_y + 1), color=color, label=f'Modelo ({label})')
    
    if is_nk:
        ax_obj.scatter(data_df['x'], np.log10(data_df['y'] + 1), label='Dados', color='black', s=20)
    else:
        df_m = data_df[data_df.type == 'mean']
        y_v = np.log10(df_m['y'] + 1)
        y_up = np.log10(data_df[data_df.type == 'up']['y'] + 1)
        y_down = np.log10(data_df[data_df.type == 'down']['y'] + 1)
        y_err = [np.abs(y_v - y_down.values), np.abs(y_up.values - y_v)]
        ax_obj.errorbar(df_m['x'], y_v, yerr=y_err, fmt='o', color=color, capsize=4, label='Dados')
    
    ax_obj.set_title(title)
    ax_obj.set_xlabel('t (dias)')
    ax_obj.set_ylabel(ylabel)
    ax_obj.grid(True, alpha=0.3)
    ax_obj.legend()

# Aplicando aos subplots
plot_data(ax[0, 0], viremia, sol.y[0], 'red', 'V', 'Vírus', r"$\log_{10}(V)$")
plot_data(ax[0, 1], il6, sol.y[14], 'blue', 'C', 'Citocinas', r"$\log_{10}(C)$")
plot_data(ax[1, 0], igg, sol.y[13], 'brown', 'IgG', 'Anticorpos IgG', r"$\log_{10}(IgG)$")
plot_data(ax[1, 1], igm, sol.y[12], 'orange', 'IgM', 'Anticorpos IgM', r"$\log_{10}(IgM)$")
plot_data(ax[2, 0], nk, sol.y[15], 'green', 'NK', 'Células NK', r"$\log_{10}(NK)$", is_nk=True)

ax[2, 1].axis('off')
plt.tight_layout()
plt.show()