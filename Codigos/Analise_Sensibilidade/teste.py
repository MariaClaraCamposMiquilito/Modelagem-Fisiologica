import numpy as np

# Número de parâmetros
D = 16

# Quantidade de simulações em cada bloco
passo = D + 2

# Separar as amostras-base da viremia
A_V = Y_V[0::passo, :]
B_V = Y_V[17::passo, :]

# Juntar as amostras-base
AB_V = np.concatenate((A_V, B_V), axis=0)

# Calcular a variância em cada instante
var_V = np.var(AB_V, axis=0)

# Repetir para as células infectadas
A_I = Y_I[0::passo, :]
B_I = Y_I[17::passo, :]

AB_I = np.concatenate((A_I, B_I), axis=0)

var_I = np.var(AB_I, axis=0)

# Mostrar os resultados
for j in range(50):
    print(
        f"Dia {t_eval[j]:.2f} | "
        f"Var(V) = {var_V[j]:.6e} | "
        f"Var(I) = {var_I[j]:.6e}"
    )