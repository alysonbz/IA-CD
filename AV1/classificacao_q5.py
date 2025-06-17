import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler  # substitua pelo melhor normalizador
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report

# 1. Carregar dados
df = pd.read_csv('dataset/classificacao_ajustado.csv')
X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis']

# 2. Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 3. Melhor configuração (exemplo: MinMaxScaler + K=7)
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=7))
])

# 4. Cross-validation com 5 folds (só treino)
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')

print(f"Acurácia média CV (5 folds): {scores.mean():.4f}")
print(f"Desvio padrão CV: {scores.std():.4f}")

# 5. Treina modelo final no treino completo e testa no conjunto teste
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# 6. Métricas finais
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("\nMatriz de Confusão:")
print(cm)

print("\nRelatório de Classificação:")
print(cr)
