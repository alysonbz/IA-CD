# Import TSNE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.utils import load_movements_price_dataset
from sklearn.preprocessing import normalize

# Cria uma instância do TSNE
model = TSNE(learning_rate='auto', init='random', random_state=42)

# Carrega e prepara os dados
movements_df = load_movements_price_dataset()
movements = movements_df.drop(['company'], axis=1)
companies = movements_df['company'].values

# Normaliza os dados
normalized_movements = normalize(movements)

# Aplica o fit_transform ao t-SNE
tsne_features = model.fit_transform(normalized_movements)

# Seleciona as duas primeiras features
xs = tsne_features[:, 0]
ys = tsne_features[:, 1]

# Gráfico de dispersão
plt.figure(figsize=(10, 6))
plt.scatter(xs, ys, alpha=0.7)

# Anota os pontos com os nomes das empresas
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)

plt.title("t-SNE - Movimentação das Ações por Empresa")
plt.xlabel("t-SNE feature 0")
plt.ylabel("t-SNE feature 1")
plt.tight_layout()
plt.show()