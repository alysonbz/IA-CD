import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.utils import load_movements_price_dataset
from sklearn.preprocessing import normalize

# Criar uma instância do TSNE
model = TSNE(learning_rate=200, random_state=0)

# Carregar os dados
movements_df = load_movements_price_dataset()
movements = movements_df.drop(['company'], axis=1)
companies = movements_df['company'].values

# Normalizar os dados
normalized_movements = normalize(movements)

# Aplicar o TSNE
tsne_features = model.fit_transform(normalized_movements)

# Selecionar as features 0 e 1
xs = tsne_features[:, 0]
ys = tsne_features[:, 1]

# Plotar gráfico de dispersão
plt.figure(figsize=(10, 6))
plt.scatter(xs, ys, alpha=0.5)

# Anotar os pontos com os nomes das empresas
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)

plt.title("t-SNE dos movimentos de preço")
plt.xlabel("t-SNE Feature 1")
plt.ylabel("t-SNE Feature 2")
plt.tight_layout()
plt.show()