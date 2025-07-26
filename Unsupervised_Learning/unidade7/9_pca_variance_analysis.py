# Perform the necessary imports
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.utils import load_fish_dataset

# Carregar o dataset
samples = load_fish_dataset()
samples = samples.drop(['specie'], axis=1)

# Criar scaler
scaler = StandardScaler()

# Criar uma instância de PCA
pca = PCA()

# Criar pipeline com scaler e PCA
pipeline = make_pipeline(scaler, pca)

# Ajustar o pipeline aos dados
pipeline.fit(samples)

# Obter variância explicada
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('Variance')
plt.xticks(features)
plt.title('Variância Explicada por Cada Componente PCA')
plt.show()