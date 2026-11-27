import numpy as np

dados = np.load("/mnt/c/Users/mique/Documents/Modelagem-Fisiologica/Codigos/Analise_Sensibilidade/Resultados_S1_40_tol/diagnostico_2e10_ln.npz")

Y_V = dados["Y_V"]
Y_I = dados["Y_I"]
t_eval = dados["t_eval"]
"""
for nome, Y in [("Viremia", Y_V), ("Infectadas", Y_I)]:

    print(f"\n{nome}")

    for j in range(50):

        valores = Y[:, j]

        # Quantas simulações estão próximas do 1e-12
        n_limite = np.sum(
            np.isclose(valores, np.log1p(1e-12), rtol=0, atol=1e-18)
        )

        # Porcentagem das simulações
        porcentagem = 100 * n_limite / len(valores)

        print(f"Dia {t_eval[j]:.2f}: {porcentagem:.2f}% | max = {np.max(valores):.6e} | var = {np.var(valores):.6e}")

"""
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

# Recuperar a escala original
Y_V_raw = np.expm1(Y_V)
Y_I_raw = np.expm1(Y_I)

D = 16
passo = D + 2

for nome, Y in [("Viremia", Y_V_raw), ("Infectadas", Y_I_raw)]:

    A = Y[0::passo, :]
    B = Y[passo - 1::passo, :]

    variancias = np.var(np.concatenate((A, B)), axis=0)

    # Limiar = 0,01% da variância máxima
    limiar = 1e-6 * np.max(variancias)
    mantidos = variancias > limiar

    print(f"\n{nome}")
    print(f"Variância máxima: {np.max(variancias):.6e}")
    print(f"Limiar: {limiar:.6e}")
    print(f"Instantes mantidos: {np.sum(mantidos)} de {len(t_eval)}")
    print("Dias descartados:", np.round(t_eval[~mantidos], 2))