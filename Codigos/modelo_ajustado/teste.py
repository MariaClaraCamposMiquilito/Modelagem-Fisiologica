import numpy as np
from modelo_covid19_ajustado import  pars

params = np.load(r'C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\modelo_ajustado\parametros_otimos_erro_quadratico.npy')
params2 = np.load(r'C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\modelo_ajustado\parametros_otimos_erro_quadratico.npy')

i = 0
for val in params2:
    print(f"{i} = {val}")
    i = i + 1
