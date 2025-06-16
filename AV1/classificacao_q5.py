# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Recarregar dataset
df = pd.read_csv('C:/Users/xulia/IA-CD/IA-CD/AV1/classificacao_ajustado.csv')
df['Medicamento'] = df['Medicamento'].astype(int)
X = pd.get_dummies(df.drop(columns=['Medicamento']), drop_first=True).astype(float)
y = df['Medicamento']

# Melhor configuração anterior: KNN com StandardScaler e k=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = KNeighborsClassifier(n_neighbors=1)

# Cross-validation
scores = cross_val_score(model, X_scaled, y, cv=5)
mean_score = scores.mean()
std_score = scores.std()

# Previsões para avaliação final
y_pred = cross_val_predict(model, X_scaled, y, cv=5)
matriz_confusao = confusion_matrix(y, y_pred)
relatorio = classification_report(y, y_pred, output_dict=False)

# Exibir resultados
print(mean_score, std_score, matriz_confusao, relatorio)