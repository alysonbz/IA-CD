# Import TSNE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.utils import load_grains_dataset

# Carrega o dataset
samples_df = load_grains_dataset()
samples = samples_df.drop(['variety', 'variety_number'], axis=1)
variety_numbers = samples_df['variety_number'].values

# Cria uma instância do TSNE
model = TSNE(learning_rate='auto', init='random', random_state=42)

# Aplica o fit_transform aos dados
tsne_features = model.fit_transform(samples)

# Seleciona as duas primeiras features
xs = tsne_features[:, 0]
ys = tsne_features[:, 1]

# Plota gráfico de dispersão colorido por variedade
plt.scatter(xs, ys, c=variety_numbers)
plt.xlabel("t-SNE feature 0")
plt.ylabel("t-SNE feature 1")
plt.title("t-SNE - Grãos")
plt.colorbar(label='Variety Number')
plt.show()