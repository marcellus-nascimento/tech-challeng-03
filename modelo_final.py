import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from fetch_data import load_datasets
from tratamento import clean_and_combine  # Importa a função que retorna o dataframe tratado
from tqdm import tqdm  # Importando a biblioteca para mostrar barras de progresso

# Caminho para a pasta onde os arquivos CSV estão localizados
save_path = r"C:\Users\Francisco\Desktop\tcfase03\techachallent"  # Defina o caminho aqui

# **Carregar os DataFrames usando load_datasets()**
print("Carregando os datasets...")
df1, df2, df3 = load_datasets(save_path)  # A função já retorna os 3 DataFrames
print("Datasets carregados com sucesso!")

# **Realizar o tratamento e combinar os dados**
print("Iniciando o tratamento e combinação dos dados...")
df = clean_and_combine(df1, df2, df3)
print("Dados tratados e combinados!")

# Separação entre treino e teste
X = df.drop(columns=["diabetes_binary"])
y = df["diabetes_binary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Definição do modelo Gradient Boosting e hiperparâmetros
param_grid = {
    "n_estimators": [100, 150],
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 5],
    "min_samples_split": [2, 5]
}

gb = GradientBoostingClassifier(random_state=42)
random_search = RandomizedSearchCV(
    gb, param_distributions=param_grid, n_iter=10, cv=5,
    scoring='accuracy', n_jobs=-1, random_state=42, verbose=2
)

print("Treinando o modelo Gradient Boosting otimizado...")
random_search.fit(X_train, y_train)

# Teste do modelo
print("Avaliando o modelo...")
y_pred = random_search.best_estimator_.predict(X_test)
y_proba = random_search.best_estimator_.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)

print(f"AUC-ROC: {roc_auc:.4f}")
print(f"F1-score: {f1:.4f}")

# Salvando o modelo treinado
modelo_filename = "gradient_boosting_model.pkl"
print(f"Salvando o modelo em {modelo_filename}...")
joblib.dump(random_search.best_estimator_, modelo_filename)
print("Modelo salvo com sucesso!")