#Redução de Dimensionalidade Múltipla: PCA vs T-SNE

#importando as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import pearsonr
import time

#Importando o csv com o pandas
df = pd.read_csv('C:/Users/xulia/IA-CD/IA-CD/AV2/mall_ajustado.csv')
print("________________________________________________________")

#criando o modelo PCA
pca_model = PCA(n_components=2)

#aplicando  o PCA
pca_features = pca_model.fit_transform(df)

pca_xs = pca_features[:, 0]
pca_ys = pca_features[:, 1]


plt.scatter(pca_xs, pca_ys)
plt.xlabel("PCA Feature 1")
plt.ylabel("PCA Feature 2")
plt.axis('equal')
plt.title("PCA - Mall Customers")
plt.show()

# Correlação entre os componentes
pca_corr, _ = pearsonr(pca_xs, pca_ys)
print("Correlação de Pearson entre PCA 1 e 2:", round(pca_corr, 3))

# Variância explicada
print("Variância explicada pelo PCA:", pca_model.explained_variance_ratio_)

#Medir tempo de execução do t-SNE
start = time.time()
tsne_model = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_features = tsne_model.fit_transform(df)
end = time.time()

tsne_xs = tsne_features[:, 0]
tsne_ys = tsne_features[:, 1]


plt.scatter(tsne_xs, tsne_ys)
plt.xlabel("t-SNE Feature 1")
plt.ylabel("t-SNE Feature 2")
plt.axis('equal')
plt.title("t-SNE - Mall Customers")
plt.show()

#correlação entre componentes t-SNE (opcional, mas geralmente baixa ou inexistente)
tsne_corr, _ = pearsonr(tsne_xs, tsne_ys)
print("Correlação de Pearson entre t-SNE 1 e 2:", round(tsne_corr, 3))

#tempo de execução
print("Tempo de execução do t-SNE: {:.2f} segundos".format(end - start))

# Comparação entre PCA e t-SNE:

# - O PCA preserva a estrutura linear e explicou aproximadamente X% da variância nos dois primeiros componentes,
#   com correlação de Pearson moderada (indicando que os componentes ainda compartilham alguma informação).
# - O t-SNE preserva relações locais e mostrou melhor separação visual entre grupos,
#   apesar de ser mais custoso computacionalmente (~Y segundos).
# - O PCA é útil para compressão, já o t-SNE é melhor para visualização de agrupamentos.
