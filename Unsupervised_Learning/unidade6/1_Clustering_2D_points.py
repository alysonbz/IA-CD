from src.utils import load_points
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Carregar os dados
points = load_points()

# Criar uma instância do KMeans com 3 clusters
model = KMeans(n_clusters=3, random_state=42)

# Separar dados de teste e treino
test_points = points[:50, :]
train_points = points[50:, :]

# Ajustar o modelo aos dados de treino
model.fit(train_points)

# Determinar os rótulos dos pontos de teste
labels = model.predict(test_points)

# Exibir os rótulos
print(labels)

# Separar coordenadas dos pontos de teste
xs = test_points[:, 0]
ys = test_points[:, 1]

# Fazer um gráfico de dispersão colorido pelos rótulos
plt.scatter(xs, ys, c=labels, cmap='viridis', alpha=0.6)

# Obter os centróides dos clusters
centroids = model.cluster_centers_

# Separar as coordenadas dos centróides
centroids_x = centroids[:, 0]
centroids_y = centroids[:, 1]

# Adicionar os centróides ao gráfico
plt.scatter(centroids_x, centroids_y, s=100, marker='D', c='red', label='Centroids')
plt.legend()
plt.title("Clusterização dos Pontos com KMeans")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()