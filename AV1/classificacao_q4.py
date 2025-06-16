import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


df = pd.read_csv(r"C:\Users\jsslu\OneDrive\Área de Trabalho\UFC\Inteligencia Artificial\git\IA-CD\AV1\datasets\voice.csv")

X = df.drop("label", axis=1)
y = df["label"]

y = y.map({'male': 1, 'female': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Use `KNeighborsClassifier` da Sklearn.
#Teste valores de `k` entre 1 e 20 e as normalizações com `GridSearchCV`.
#Plote gráfico dos resultados com as 3 melhores configurações.


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parâmetros
param_grid = {
    'knn__n_neighbors': list(range(1, 21))
}

# Dicionário para guardar os resultados
resultados = {}

# Normalizações
normalizacoes = {
    'Log': np.log1p,
    'MinMax': MinMaxScaler(),
    'Standard': StandardScaler()
}

# Avaliação de cada normalização com GridSearch
for nome, normalizador in normalizacoes.items():
    print(f"Testando normalização: {nome}")

    if nome == 'Log':
        X_train_norm = normalizador(X_train)
        X_test_norm = normalizador(X_test)
        model = KNeighborsClassifier()
        grid = GridSearchCV(estimator=model,
                            param_grid={'n_neighbors': list(range(1, 21))},
                            cv=5)
        grid.fit(X_train_norm, y_train)
        best_score = grid.best_score_
        best_k = grid.best_params_['n_neighbors']
        acc_test = accuracy_score(y_test, grid.predict(X_test_norm))
    else:
        pipeline = Pipeline([
            ('scaler', normalizador),
            ('knn', KNeighborsClassifier())
        ])
        grid = GridSearchCV(pipeline, param_grid=param_grid, cv=5)
        grid.fit(X_train, y_train)
        best_score = grid.best_score_
        best_k = grid.best_params_['knn__n_neighbors']
        acc_test = accuracy_score(y_test, grid.predict(X_test))

    resultados[nome] = {
        'best_k': best_k,
        'val_accuracy': best_score,
        'test_accuracy': acc_test
    }

# Ordenar pelos melhores
melhores = sorted(resultados.items(), key=lambda x: x[1]['val_accuracy'], reverse=True)[:3]

# Plotar
labels = [x[0] for x in melhores]
val_scores = [x[1]['val_accuracy'] for x in melhores]
test_scores = [x[1]['test_accuracy'] for x in melhores]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, val_scores, width, label='Validação (CV)')
rects2 = ax.bar(x + width / 2, test_scores, width, label='Teste')

ax.set_ylabel('Acurácia')
ax.set_title('Top 3 Normalizações com melhor K')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.ylim(0.85, 1.0)

for i in range(len(labels)):
    plt.text(x[i] - 0.25, val_scores[i] + 0.005, f"{val_scores[i]:.2f}")
    plt.text(x[i] + 0.05, test_scores[i] + 0.005, f"{test_scores[i]:.2f}")

plt.tight_layout()
plt.show()

melhor_nome, melhor_resultado = melhores[0]
print(f"\nMelhor configuração encontrada:")
print(f"Normalização: {melhor_nome}")
print(f"Melhor K: {melhor_resultado['best_k']}")
print(f"Acurácia na Validação (CV): {melhor_resultado['val_accuracy']:.4f}")
print(f"Acurácia no Teste: {melhor_resultado['test_accuracy']:.4f}")