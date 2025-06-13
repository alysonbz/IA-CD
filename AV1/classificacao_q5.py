import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar dados
df = pd.read_csv('classificacao_ajustado.csv')
df = df.drop(columns='id')
X = df.drop(columns='label')
y = df['label']

log_transformer = FunctionTransformer(
    func=lambda x: np.log1p(np.clip(x, a_min=1e-5, a_max=None)), validate=True
)

pipeline = Pipeline([
    ('scaler', log_transformer),
    ('knn', KNeighborsClassifier(n_neighbors=20))
])

# 5-fold Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(pipeline, X, y, cv=cv)
print(f"Acurácia média: {scores.mean():.4f}")
print(f"Desvio padrão: {scores.std():.4f}")

# Previsões para matriz de confusão e relatório
y_pred = cross_val_predict(pipeline, X, y, cv=cv)

# Matriz de confusão
conf_matrix = confusion_matrix(y, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusão - KNN com Cross-Validation (k=20, log)")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y, y_pred))
