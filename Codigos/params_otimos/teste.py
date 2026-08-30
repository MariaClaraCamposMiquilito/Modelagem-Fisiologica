import numpy as np

params = np.load(r'C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\params_otimos\parametros_otimos.npy')

i = 0
for val in params:
    print(f"{i} = {val}")
    i = i + 1
