from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time

# Selecionar variáveis numéricas
X = df[['math score', 'reading score', 'writing score']]

# Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- PCA ----
pca = PCA(n_components=2)

start_pca = time.time()
X_pca = pca.fit_transform(X_scaled)
end_pca = time.time()

explained_variance = pca.explained_variance_ratio_

# ---- t-SNE ----
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)

start_tsne = time.time()
X_tsne = tsne.fit_transform(X_scaled)
end_tsne = time.time()

# ---- Visualizações ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PCA Scatter
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], alpha=0.7, ax=axes[0])
axes[0].set_title(f'PCA (Var Exp: {explained_variance.sum():.2%})')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

# t-SNE Scatter
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], alpha=0.7, ax=axes[1])
axes[1].set_title('t-SNE Projection')
axes[1].set_xlabel('Dim 1')
axes[1].set_ylabel('Dim 2')

plt.tight_layout()
plt.show()

explained_variance, (end_pca - start_pca), (end_tsne - start_tsne)
