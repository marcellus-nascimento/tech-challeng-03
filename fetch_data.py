import os
import pandas as pd

def load_datasets(path):
    """Lê os arquivos CSV de uma pasta local e retorna os dataframes."""
    files = os.listdir(path)
    print("Arquivos disponíveis:", files)

    # Carregue os arquivos CSV diretamente da pasta
    df1 = pd.read_csv(os.path.join(path, files[2]))
    df2 = pd.read_csv(os.path.join(path, files[3]))
    df3 = pd.read_csv(os.path.join(path, files[4]))

    print(f"\nDataframe: {files[0]} - Tamanho: {df1.shape}")
    print(f"\nDataframe: {files[1]} - Tamanho: {df2.shape}")
    print(f"\nDataframe: {files[2]} - Tamanho: {df3.shape}")

    return df1, df2, df3

# Caminho para a pasta onde os arquivos CSV estão localizados
save_path = r"C:\Users\Francisco\Desktop\tcfase03\techachallent"

# Carrega os datasets diretamente da pasta
df1, df2, df3 = load_datasets(save_path)
