import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import time

# 1. Carregar e limpar dados
df = pd.read_excel('./AV2/dados/dados.xlsx')
df = df.dropna(subset=['CustomerID', 'Description'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# 2. Criar atributo de valor total
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# 3. Selecionar atributos numéricos
features = ['Quantity', 'UnitPrice', 'TotalPrice']
X = df[features]

# 4. Padronização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. PCA
pca = PCA(n_components=2)
start_pca = time.time()
X_pca = pca.fit_transform(X_scaled)
end_pca = time.time()
pca_time = end_pca - start_pca

# 6. T-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
start_tsne = time.time()
X_tsne = tsne.fit_transform(X_scaled)
end_tsne = time.time()
tsne_time = end_tsne - start_tsne

# 7. Visualização comparativa
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# PCA plot
axs[0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, s=10, c='blue')
axs[0].set_title(f'PCA - Tempo: {pca_time:.2f} segundos')

# T-SNE plot
axs[1].scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5, s=10, c='green')
axs[1].set_title(f'T-SNE - Tempo: {tsne_time:.2f} segundos')

for ax in axs:
    ax.set_xlabel('Componente 1')
    ax.set_ylabel('Componente 2')
    
plt.suptitle('Redução de Dimensionalidade: PCA vs T-SNE')
plt.tight_layout()
plt.tight_layout()
plt.savefig('comparacao_pca_tsne.png')
print("Gráfico salvo como 'comparacao_pca_tsne.png'")

# 8. Variância explicada do PCA
print("Variância explicada pelo PCA:", pca.explained_variance_ratio_)
