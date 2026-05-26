import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler
)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# 1. CARREGAR DATASET
# =========================================================

df = pd.read_csv('adult/adult.data', header=None, na_values='?')

df.columns = [
    'age','workclass','fnlwgt','education','education-num',
    'marital-status','occupation','relationship','race',
    'sex','capital-gain','capital-loss','hours-per-week',
    'native-country','income'
]

print(df.head())
print(df.info())

# =========================================================
# 2. TRATAMENTO DE DADOS
# =========================================================

df.dropna(inplace=True)

# Variável alvo
y = df['income']

# Variáveis de entrada
X = df.drop('income', axis=1)

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

print("\nShape final:", X.shape)

# =========================================================
# 3. ANÁLISE DOS ATRIBUTOS
# =========================================================

plt.figure(figsize=(12,8))

corr = X.corr()

sns.heatmap(corr, cmap='coolwarm')

plt.title("Heatmap de Correlação")
plt.savefig("heatmap_classificacao.png")
plt.close()

# =========================================================
# 4. DIVISÃO TREINO/TESTE
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# =========================================================
# 5. NORMALIZAÇÕES
# =========================================================

# ---------- LOG ----------
X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)

# ---------- MINMAX ----------
scaler_mm = MinMaxScaler()

X_train_mm = scaler_mm.fit_transform(X_train)
X_test_mm = scaler_mm.transform(X_test)

# ---------- Z-SCORE ----------
scaler_z = StandardScaler()

X_train_z = scaler_z.fit_transform(X_train)
X_test_z = scaler_z.transform(X_test)

# =========================================================
# 6. TESTE DE K
# =========================================================

ks = range(1, 31)

acc_log = []
acc_mm = []
acc_z = []

for k in ks:

    # LOG
    model_log = KNeighborsClassifier(n_neighbors=k)

    model_log.fit(X_train_log, y_train)

    pred_log = model_log.predict(X_test_log)

    acc_log.append(
        accuracy_score(y_test, pred_log)
    )

    # MINMAX
    model_mm = KNeighborsClassifier(n_neighbors=k)

    model_mm.fit(X_train_mm, y_train)

    pred_mm = model_mm.predict(X_test_mm)

    acc_mm.append(
        accuracy_score(y_test, pred_mm)
    )

    # Z-SCORE
    model_z = KNeighborsClassifier(n_neighbors=k)

    model_z.fit(X_train_z, y_train)

    pred_z = model_z.predict(X_test_z)

    acc_z.append(
        accuracy_score(y_test, pred_z)
    )

# =========================================================
# 7. GRÁFICO K x ACURÁCIA
# =========================================================

plt.figure(figsize=(10,6))

plt.plot(ks, acc_log, marker='o', label='Log')
plt.plot(ks, acc_mm, marker='o', label='MinMax')
plt.plot(ks, acc_z, marker='o', label='Z-score')

plt.xlabel("Valor de K")
plt.ylabel("Acurácia")

plt.title("KNN - K x Acurácia")

plt.legend()

plt.grid(True)

plt.savefig("grafico_knn.png")
plt.close()

# =========================================================
# 8. MELHOR MODELO
# =========================================================

melhor_k = ks[acc_z.index(max(acc_z))]

print(f"\nMelhor K (Z-score): {melhor_k}")
print(f"Melhor acurácia: {max(acc_z):.4f}")

# =========================================================
# 9. MODELO FINAL
# =========================================================

modelo_final = KNeighborsClassifier(
    n_neighbors=melhor_k
)

modelo_final.fit(X_train_z, y_train)

y_pred = modelo_final.predict(X_test_z)

# =========================================================
# 10. MATRIZ DE CONFUSÃO
# =========================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Matriz de Confusão")

plt.xlabel("Previsto")
plt.ylabel("Real")

plt.savefig("matriz_confusao.png")
plt.close()

# =========================================================
# 11. CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# =========================================================
# 12. VALIDAÇÃO CRUZADA
# =========================================================

scores_cv = cross_val_score(
    KNeighborsClassifier(n_neighbors=melhor_k),
    X_train_z,
    y_train,
    cv=5
)

print("\nValidação Cruzada:")
print(scores_cv)

print("Média:", scores_cv.mean())

# =========================================================
# 13. GRID SEARCH
# =========================================================

param_grid = {
    'n_neighbors': ks
}

grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid.fit(X_train_z, y_train)

print("\nMelhores parâmetros:")

print(grid.best_params_)

print("Melhor score:")

print(grid.best_score_)

# =========================================================
# 14. COMPARAÇÃO DAS NORMALIZAÇÕES
# =========================================================

resultados = {
    "Log": max(acc_log),
    "MinMax": max(acc_mm),
    "Z-score": max(acc_z)
}

plt.figure(figsize=(6,5))

plt.bar(
    resultados.keys(),
    resultados.values()
)

plt.ylabel("Acurácia")

plt.title("Comparação das Normalizações")

plt.savefig("comparacao_normalizacao.png")
plt.close()

print("\nResultados das normalizações:")

for nome, valor in resultados.items():
    print(nome, "=", round(valor,4))