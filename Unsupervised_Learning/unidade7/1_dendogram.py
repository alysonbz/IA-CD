import matplotlib.pyplot as plt
from src.utils import load_grains_splited_datadet

# Import linkage and dendrogram
from scipy.cluster.hierarchy import linkage, dendrogram

# Carregar os dados
X_train, samples, y_train, varieties = load_grains_splited_datadet()

# Calcular a ligação (linkage): mergings
mergings = linkage(X_train, method='complete')  # ou 'ward', 'single', etc.

# Plotar o dendrograma, usando varieties como rótulos
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=10,
)
plt.show()
