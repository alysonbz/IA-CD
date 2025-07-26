# Perform the necessary imports
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.utils import load_fish_dataset

# Carrega o dataset
samples = load_fish_dataset()
samples = samples.drop(['specie'], axis=1)

# Cria os componentes do pipeline
scaler = StandardScaler()
pca = PCA(n_components=2)

# Cria o pipeline corretamente
pipeline = make_pipeline(scaler, pca)

# Ajusta o pipeline aos dados
pipeline.fit(samples)

# Acessa o PCA dentro do pipeline (último passo)
pca_fitted = pipeline.named_steps['pca']

# Plota a variância explicada
features = range(pca.n_components)
plt.bar(features, pca_fitted.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()