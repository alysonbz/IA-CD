#t-SNE ou PCA como Pré-processamento para Classificação

# Importando bibliotecas
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Carregando o dataset
df = pd.read_csv('C:/Users/xulia/IA-CD/IA-CD/AV2/mall_ajustado.csv')

# Separando variáveis independentes e o alvo (cluster gerado pelo K-Means)
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
y = df['cluster']  # alvo de classificação

#Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Divisão treino/teste (para manter justo entre os métodos)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

#Classificação com dados apenas normalizados

clf_norm = RandomForestClassifier(random_state=42)
clf_norm.fit(X_train, y_train)
y_pred_norm = clf_norm.predict(X_test)
acc_norm = accuracy_score(y_test, y_pred_norm)

#Classificação com PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

clf_pca = RandomForestClassifier(random_state=42)
clf_pca.fit(X_train_pca, y_train)
y_pred_pca = clf_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)

#Classificação com t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)
X_train_tsne, X_test_tsne, y_train, y_test = train_test_split(X_tsne, y, test_size=0.3, random_state=42)

clf_tsne = RandomForestClassifier(random_state=42)
clf_tsne.fit(X_train_tsne, y_train)
y_pred_tsne = clf_tsne.predict(X_test_tsne)
acc_tsne = accuracy_score(y_test, y_pred_tsne)

#Resultados
print("Acurácias dos modelos:")
print(f"1. Normalização apenas: {acc_norm:.4f}")
print(f"2. Com PCA (2D):        {acc_pca:.4f}")
print(f"3. Com t-SNE (2D):      {acc_tsne:.4f}")

# Conclusão:

# Foram avaliados três cenários de classificação para prever os grupos formados pelo K-Means:
# 1. Dados apenas normalizados (StandardScaler) → acurácia de 91,67%
# 2. Redução de dimensionalidade com PCA → acurácia de 95,00%
# 3. Redução com t-SNE → acurácia de 93,33%
# O melhor desempenho foi obtido com o uso do PCA, que além de reduzir as dimensões para 2,
# conseguiu preservar a estrutura global dos dados e melhorar a performance do classificador.
# O uso de t-SNE também melhorou o desempenho em relação à normalização sozinha,
# mostrando que a projeção não-linear foi capaz de capturar padrões relevantes para a classificação.
# A clusterização auxiliou a definir um alvo realista e revelou padrões que foram bem aprendidos
# pelos modelos supervisionados, validando a integração entre técnicas não supervisionadas e supervisionadas.
# Concluímos que o PCA pode ser uma excelente estratégia de pré-processamento,
# especialmente quando buscamos reduzir dimensionalidade sem perder desempenho.
