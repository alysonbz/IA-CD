# Import PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from src.utils import load_grains_dataset

# Carregar o dataset
grains = load_grains_dataset()
grains = grains.drop(['variety', 'variety_number'], axis=1)

# Criar instância do PCA
model = PCA()

# Aplicar o PCA nos dados
pca_features = model.fit_transform(grains)

# Atribuir os componentes principais
xs = pca_features[:, 0]
ys = pca_features[:, 1]

# Gráfico de dispersão entre os dois primeiros componentes
plt.scatter(xs, ys)
plt.axis('equal')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('PCA - Grãos')
plt.show()

# Calcular a correlação de Pearson entre os dois componentes
correlation, pvalue = pearsonr(xs, ys)

# Exibir a correlação
print("Correlação de Pearson entre PCA 1 e 2:", round(correlation, 3))