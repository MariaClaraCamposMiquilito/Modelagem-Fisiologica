import pandas as pd
import os

pasta_entrada = r'C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\meus_dados\Dados_Novos'
pasta_saida = r'C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\meus_dados\Dados_Corrigidos'

os.makedirs(pasta_saida, exist_ok=True)

for arquivo in os.listdir(pasta_entrada):
    if arquivo.endswith(".csv"):
        caminho_arquivo = os.path.join(pasta_entrada, arquivo)

        print(f"Processando: {arquivo}")

        # Lê com separador original ";"
        df = pd.read_csv(caminho_arquivo, sep=';', header=None)

        # Corrige decimal nas duas primeiras colunas
        for col in [0, 1]:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)

        # Primeira coluna como inteiro arredondado
        df[0] = pd.to_numeric(df[0], errors='coerce').round().astype(int)

        # Remove espaços extras
        df = df.applymap(lambda x: str(x).strip())

        # Define o cabeçalho
        df.columns = ["x", "y", "tuple", "type"]
        df["y"] = pd.to_numeric(df["y"], errors='coerce') / 1000


        # Salva com separador "," e cabeçalho
        caminho_saida = os.path.join(pasta_saida, arquivo)
        df.to_csv(caminho_saida, index=False, sep=',')

        print(f"{arquivo} corrigido!")

print("Finalizado!")