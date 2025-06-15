import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# --------------------
# 1. Ler e preparar os dados
# --------------------
df = pd.read_csv("IA-CD/AV1/bancos/flavors_of_cacao.csv")
df.columns = [
    "Company", "Specific_Bean_Origin", "REF", "Review_Date", "Cocoa_Percent",
    "Company_Location", "Rating", "Bean_Type", "Broad_Bean_Origin"
]
# Limpar nomes de colunas
df.columns = [col.strip() for col in df.columns]

# Criar variável alvo
df['target'] = (df['Rating'] >= 3.0).astype(int)

# Remover colunas irrelevantes e tratar categóricas
df = df.dropna()
X = df.drop(columns=['Company', 'Specific_Bean_Origin', 'REF', 'Review_Date', 'Cocoa_Percent', 'Company_Location', 'Rating', 'target'])
X = pd.get_dummies(X)
y = df['target']

# --------------------
# 2. Melhor configuração encontrada (StandardScaler + k=17)
# --------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=17)

# --------------------
# 3. Aplicar Cross-validation (5 folds)
# --------------------
scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
print(f"\nMédia da Acurácia (5 folds): {scores.mean():.4f}")
print(f"Desvio Padrão da Acurácia: {scores.std():.4f}")

# --------------------
# 4. Previsões para Matriz de Confusão e Relatório de Classificação
# --------------------
y_pred = cross_val_predict(knn, X_scaled, y, cv=5)

cm = confusion_matrix(y, y_pred)
print("\nMatriz de Confusão:")
print(cm)

print("\nRelatório de Classificação:")
print(classification_report(y, y_pred))
