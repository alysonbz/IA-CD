import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Carregar dados
df = pd.read_csv('dataset/marketing_campaign_preprocessed.csv')
df = df.drop(columns=['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain', 'Response'], errors='ignore')

# Padronizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(pca_result, columns=["PCA1", "PCA2"])

# T-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(X_scaled)
tsne_df = pd.DataFrame(tsne_result, columns=["TSNE1", "TSNE2"])

# Visualizações
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", ax=axs[0], alpha=0.7)
axs[0].set_title("PCA")
sns.scatterplot(data=tsne_df, x="TSNE1", y="TSNE2", ax=axs[1], alpha=0.7)
axs[1].set_title("T-SNE")
plt.suptitle("Comparação entre PCA e T-SNE", fontsize=16)
plt.tight_layout()
plt.show()

# Variância explicada PCA
print("Variância explicada:", pca.explained_variance_ratio_)
print("Variância total:", sum(pca.explained_variance_ratio_))
