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
print("Dividindo os dados entre treino e teste...")
X = df.drop(columns=["diabetes_binary"])
y = df["diabetes_binary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Divisão concluída!")

# Definição dos modelos e hiperparâmetros
models_params = {
    "Random Forest": (RandomForestClassifier(random_state=42), {
        "n_estimators": [100, 150, 200], "max_depth": [10, 15, None],
        "min_samples_split": [2, 5], "min_samples_leaf": [1, 2]
    }),
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), {
        "C": [0.1, 1, 10], "solver": ['liblinear'], "max_iter": [500]
    }),
    "Decision Tree": (DecisionTreeClassifier(random_state=42), {
        "max_depth": [5, 10, 15, None], "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }),
    "KNN": (KNeighborsClassifier(), {
        "n_neighbors": [3, 5, 7], "weights": ['uniform', 'distance']
    }),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=42), {
        "n_estimators": [100, 150], "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5], "min_samples_split": [2, 5]
    })
}

best_model, best_model_name, best_score = None, None, 0

# Treinamento e avaliação com validação cruzada usando tqdm
print("Iniciando a validação cruzada...")
for name, (model, _) in tqdm(models_params.items(), desc="Treinando Modelos", unit="modelo"):
    print(f"\nTreinando o modelo: {name}...")
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    mean_score = np.mean(scores)
    print(f"{name}: Acurácia média (validação cruzada) = {mean_score:.4f}")
    if mean_score > best_score:
        best_model, best_model_name, best_score = model, name, mean_score

print(f"\nMelhor modelo: {best_model_name} com acurácia média de {best_score:.4f}")

# Otimização de hiperparâmetros com RandomizedSearchCV usando tqdm
print(f"Iniciando a otimização de hiperparâmetros para {best_model_name}...")
random_search = RandomizedSearchCV(
    best_model, param_distributions=models_params[best_model_name][1],
    n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42, verbose=2
)
random_search.fit(X_train, y_train)

# Teste do modelo otimizado
y_pred_optimized = random_search.best_estimator_.predict(X_test)
roc_auc = roc_auc_score(y_test, y_pred_optimized)
f1 = f1_score(y_test, y_pred_optimized)

print(f"\nMelhores parâmetros para {best_model_name}: {random_search.best_params_}")
print(f"Melhor acurácia após otimização: {random_search.best_score_:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")
print(f"F1-score: {f1:.4f}")

# Balanceamento e Ensemble Learning com tqdm
print("\nIniciando o balanceamento e o ensemble learning...")
# Balanceamento e Ensemble Learning
dt = DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=2,
                            class_weight='balanced', random_state=42)
dt.fit(X_train, y_train)  # A chamada do fit estava aqui
ensemble = VotingClassifier(
    estimators=[('Decision Tree', dt), ('Gradient Boosting', random_search.best_estimator_)],
    voting='soft'
)
ensemble.fit(X_train, y_train)


ensemble = VotingClassifier(
    estimators=[('Decision Tree', dt), ('Gradient Boosting', random_search.best_estimator_)],
    voting='soft'
)

print("Treinando o modelo Ensemble...")
ensemble.fit(X_train, y_train)

# Treinando o modelo ensemble

for _ in tqdm(range(1), desc="Treinando Ensemble", unit="iter"):
    ensemble.fit(X_train, y_train)

# Avaliação final
models = {
    "Gradient Boosting (Otimizado)": random_search.best_estimator_,
    "Decision Tree (Balanceado)": dt,
    "Ensemble (Voting Classifier)": ensemble
}

print("\nAvaliação final dos modelos:")
for name, model in tqdm(models.items(), desc="Avaliação Final", unit="modelo"):
    print(f"\nTreinando o modelo {name}...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    print(f"{name} - AUC-ROC: {auc:.4f}")
    print(f"{name} - F1-score: {f1:.4f}")
    print(classification_report(y_test, y_pred))


