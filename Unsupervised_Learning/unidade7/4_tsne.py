import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.utils import load_grains_dataset

# Carregar os dados
samples_df = load_grains_dataset()
samples = samples_df.drop(['variety', 'variety_number'], axis=1)
variety_numbers = samples_df['variety_number'].values

# Criar uma instância do TSNE
model = TSNE(learning_rate=200, random_state=0)

# Aplicar fit_transform aos dados
tsne_features = model.fit_transform(samples)

# Selecionar a primeira e segunda dimensão do TSNE
xs = tsne_features[:, 0]
ys = tsne_features[:, 1]

# Plotar gráfico de dispersão colorido pelos rótulos das variedades
plt.scatter(xs, ys, c=variety_numbers)
plt.xlabel("TSNE Feature 1")
plt.ylabel("TSNE Feature 2")
plt.title("Visualização com t-SNE")
plt.colorbar(label="Variety Number")

plt.show()