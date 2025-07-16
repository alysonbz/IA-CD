# 8
# Import PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from src.utils import load_grains_dataset

# Carregar os dados e remover colunas de rótulo
grains = load_grains_dataset()
grains = grains.drop(['variety', 'variety_number'], axis=1)

# Criar uma instância do PCA
model = PCA()

# Aplicar o PCA aos dados
pca_features = model.fit_transform(grains)

# Atribuir os componentes principais
xs = pca_features[:, 0]
ys = pca_features[:, 1]

# Gráfico de dispersão dos dois primeiros componentes
plt.scatter(xs, ys)
plt.xlabel("PCA Feature 1")
plt.ylabel("PCA Feature 2")
plt.axis('equal')
plt.title("PCA - Grãos")
plt.show()

# Calcular a correlação de Pearson entre os componentes
correlation, pvalue = pearsonr(xs, ys)

# Exibir a correlação
print("Correlação de Pearson entre PCA 1 e 2:", correlation)
