import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados (ajuste o caminho se necessário)

df = pd.read_excel('./AV2/dados/dados.xlsx')
# Limpeza
df = df.dropna()
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Agrupamento por cliente
df_grouped = df.groupby(['CustomerID', 'Country']).agg({
    'Quantity': 'sum',
    'UnitPrice': 'mean'
}).reset_index()

# Normalização
features = df_grouped[['Quantity', 'UnitPrice']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# K-Means - escolha do melhor k
inertias = []
silhouettes = []
k_range = range(2, 7)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

best_k = k_range[silhouettes.index(max(silhouettes))]
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Hierárquico
agg_avg = AgglomerativeClustering(n_clusters=best_k, linkage='average')
avg_labels = agg_avg.fit_predict(X_scaled)

agg_complete = AgglomerativeClustering(n_clusters=best_k, linkage='complete')
complete_labels = agg_complete.fit_predict(X_scaled)

# PCA para visualização
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# DataFrame para plotagem
df_vis = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_vis['KMeans'] = kmeans_labels
df_vis['AvgLink'] = avg_labels
df_vis['CompleteLink'] = complete_labels

# Visualizações
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.scatterplot(data=df_vis, x='PC1', y='PC2', hue='KMeans', ax=axes[0]).set_title('K-Means')
sns.scatterplot(data=df_vis, x='PC1', y='PC2', hue='AvgLink', ax=axes[1]).set_title('Hierárquico - Average')
sns.scatterplot(data=df_vis, x='PC1', y='PC2', hue='CompleteLink', ax=axes[2]).set_title('Hierárquico - Complete')
plt.tight_layout()
plt.show()

# Métricas de avaliação
print("Melhor k:", best_k)
print("Inércias:", inertias)
print("Silhouette Scores:", silhouettes)
