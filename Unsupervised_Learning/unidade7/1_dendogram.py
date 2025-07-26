import matplotlib.pyplot as plt
from src.utils import load_grains_splited_datadet

# Import linkage and dendrogram
from scipy.cluster.hierarchy import linkage, dendrogram

# Carregar os dados
X_train, samples, y_train, varieties = load_grains_splited_datadet()

# Calcular as ligações para o dendrograma
mergings = linkage(X_train, method='complete')

# Plotar o dendrograma com os rótulos das variedades
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=10,
)
plt.title("Dendrograma de Clusterização Hierárquica - Grãos")
plt.xlabel("Amostras")
plt.ylabel("Distância")
plt.tight_layout()
plt.show()