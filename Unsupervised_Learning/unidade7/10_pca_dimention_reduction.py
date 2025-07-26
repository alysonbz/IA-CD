from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.utils import load_fish_dataset
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Carregar o dataset
samples = load_fish_dataset()
samples = samples.drop(['specie'], axis=1)

# Padronizar os dados
scaler = StandardScaler()
scaled_samples = scaler.fit_transform(samples)

# Criar o modelo PCA com 2 componentes
pca = PCA(n_components=2)

# Ajustar o PCA aos dados padronizados
pca.fit(scaled_samples)

# Transformar os dados com PCA
pca_features = pca.transform(scaled_samples)

# Mostrar a forma dos dados transformados
print("Shape dos dados após PCA:", pca_features.shape)

# Visualizar os dados reduzidos em gráfico de dispersão
plt.scatter(pca_features[:, 0], pca_features[:, 1])
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('PCA - Redução de Dimensão (Fish Dataset)')
plt.axis('equal')
plt.show()