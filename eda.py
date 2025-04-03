import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tratamento import clean_and_combine
from fetch_data import load_datasets

# Configuração de estilo para os gráficos
sns.set_style("whitegrid")

# Caminho da pasta onde os arquivos CSV estão localizados
save_path = r"C:\Users\Francisco\Desktop\tcfase03\techachallent"

# Carregando os datasets
df1, df2, df3 = load_datasets(save_path)

# Limpando e combinando os DataFrames
df = clean_and_combine(df1, df2, df3)

# Função de análise exploratória
print("\nResumo Estatístico:")
print(df.describe())

print("\nDistribuição das Classes:")
print(df["diabetes_binary"].value_counts())

# Visualização da distribuição das classes
plt.figure(figsize=(6, 4))
ax = sns.countplot(x="diabetes_binary", data=df, palette="coolwarm")

# Adicionando rótulos nas barras
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha="center", va="bottom", fontsize=8, color="black", fontweight="bold")

# Ajustando rótulos e título
plt.title("Distribuição da variável alvo (Diabetes_binary)")
plt.xlabel("Diabetes_binary")
plt.ylabel("Quantidade")
plt.xticks(ticks=[0, 1], labels=["Não", "Sim"])
plt.show()

# Matriz de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlação entre as Features")
plt.show()


def plot_histograms(df):
    """Gera histogramas para todas as colunas numéricas do DataFrame."""
    num_cols = df.select_dtypes(include=["number"]).columns  # Apenas colunas numéricas
    num_features = len(num_cols)

    plt.figure(figsize=(12, num_features * 3))  # Ajustar o tamanho da figura

    for i, col in enumerate(num_cols, 1):
        plt.subplot((num_features // 3) + 1, 3, i)  # Criar subplots dinâmicos
        sns.histplot(df[col], bins=20, kde=True, color="royalblue", edgecolor="black")

        # Adicionar linha da média e mediana
        mean = df[col].mean()
        median = df[col].median()
        plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Média: {mean:.2f}')
        plt.axvline(median, color='green', linestyle='dashed', linewidth=2, label=f'Mediana: {median:.2f}')
        plt.legend()

        plt.title(f"Histograma de {col}")
        plt.xlabel(col)
        plt.ylabel("Frequência")

    plt.tight_layout()
    plt.show()


# Chamar a função para gerar histogramas
plot_histograms(df)