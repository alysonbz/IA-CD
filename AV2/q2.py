from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


#padronização dos dados
X = cliente_df[['NumCompras', 'TotalItens', 'PrecoMedio', 'Recencia']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("PCA")
plt.show()

# T-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure()
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.title("T-SNE")
plt.show()

# Comparativo: PCA preserva variância, T-SNE foca em agrupamento local

