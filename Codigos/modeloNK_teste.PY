import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# DEFINIÇÃO DOS PARÂMETROS (Cenário Severo)
beta_1 = 0.5
beta_2 = 1.0
gamma_1 = 30.0

# CONSTANTES 
tau_c = 6.0/24.0
tau_n = 24.0/24.0
tau_m = 100.0/24.0
k1 = 10**5
k2 = 5.0
gamma_2 = 15.0
e = 0.05

# CONDIÇÕES INICIAIS
Na0 = 0.0
Ma0 = 0.0
Mm0 = 0.0
V0 = 1.0

y0 = [Na0, Ma0, Mm0, V0] 
tempo = np.linspace(0, 20, 200)

# FUNÇÃO DO MODELO (SISTEMA DE EDOs)
def modelo_nk(y, t, k1, k2, tau_c, tau_n, tau_m, e, beta_1, beta_2, gamma_1, gamma_2):
    Na, Ma, Mm, V = y
    
    IFN_I = ((beta_1 * V) / (V + k1)) + ((beta_2 * Ma) / (Ma + k2))
    IL_6 = ((gamma_1 * V) / (V + k1)) + ((gamma_2 * Ma) / (Ma + k2))
    IFN_gamma = Na + (Ma / (Ma + k2)) 

    dVdt = ((1.0 / tau_c) * (1 - Na - Ma) * V)
    dNadt = ((1.0 / tau_n) * (IFN_I - Na))
    dMmdt = ((1.0 / tau_m) * (IL_6 - Mm))
    dMadt = ((1.0 / tau_m) * (e + (Mm * IFN_gamma) - Ma))

    return [dNadt, dMadt, dMmdt, dVdt]

# Argumentos para a função modelo_nk (passados ao ODEINT)
args_modelo_nk = (k1, k2, tau_c, tau_n, tau_m, e, beta_1, beta_2, gamma_1, gamma_2)

# ODEINT
solucao = odeint(modelo_nk, y0, tempo, args=args_modelo_nk)
Na = solucao[:, 0]
Ma = solucao[:, 1]
Mm = solucao[:, 2]
V = solucao[:, 3]

# PLOTANDO O GRÁFICO DAS NK
plt.figure(figsize=(10, 6))
plt.plot(tempo, Na, label = 'Células NK ativadas')
plt.title('Dinâmica das Células NK Ativadas')
plt.ylabel('Nível de ativação')
plt.xlabel('Dias após a infecção')
plt.scatter(0,0 , marker='o', color = 'red', label = 'dados')
plt.ylim(0, 2) 
plt.xlim(0, 20)
plt.grid(True, linestyle='--')
plt.legend()
plt.show()