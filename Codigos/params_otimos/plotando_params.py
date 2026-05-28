import numpy as np

params_otimos = np.load(r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\params_otimos\parametros_otimos.npy")
cont = 0

for i in params_otimos:
    print(f"{cont}: {i}")
    cont = cont + 1