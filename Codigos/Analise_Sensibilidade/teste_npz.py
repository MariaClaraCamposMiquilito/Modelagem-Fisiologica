import numpy as np

dados = np.load("/mnt/c/Users/mique/Documents/Modelagem-Fisiologica/Codigos/Analise_Sensibilidade/Resultados_S1_40_tol/diagnostico_2e10_ln.npz")

Y_V = dados["Y_V"]
Y_I = dados["Y_I"]
t_eval = dados["t_eval"]

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

