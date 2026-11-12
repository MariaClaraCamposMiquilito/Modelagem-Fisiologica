import numpy as np
import matplotlib.pyplot as plt

# Carrega o arquivo salvo (exemplo para 2^15)
dados = np.load("/home/user/Maria_Clara/Modelagem-Fisiologica/Codigos/Analise_Sensibilidade/Resultados_S2_testes/resultados_sobol_2e15_bounds_novos.npz")

t_eval = dados["t_eval"]
Y_virus = dados["Y_Virus"]
Y_infectadas = dados["Y_Infectadas"]

# Calcula a variância total ao longo das simulações (axis=0) para cada tempo
var_virus = np.var(Y_virus, axis=0, ddof=1)
var_infectadas = np.var(Y_infectadas, axis=0, ddof=1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t_eval, var_virus, color='blue', lw=2)
plt.axvline(x=25, color='red', linestyle='--', label='t ≈ 25')
plt.title('Variância Total - Vírus')
plt.xlabel('Tempo')
plt.ylabel('Var(Y)')
plt.grid(True, which='both', ls='--')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t_eval, var_infectadas, color='orange', lw=2)
plt.axvline(x=25, color='red', linestyle='--', label='t ≈ 25')
plt.title('Variância Total - Células Infectadas')
plt.xlabel('Tempo')
plt.ylabel('Var(Y)')
plt.grid(True, which='both', ls='--')
plt.legend()

plt.tight_layout()
plt.show()