import pandas as pd

pop_moderado_up = pd.read_csv(r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\meus_dados\NK_covid_critico.csv")
por_moderado_mean = pd.read_csv(r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\meus_dados\NK_covid_critico.csv")
pop_moderado_down = pd.read_csv(r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\meus_dados\NK_covid_critico.csv")


plt.plot(nk_critico[['x'].values], nk_critico['y'].values, label = "NK Crítico")