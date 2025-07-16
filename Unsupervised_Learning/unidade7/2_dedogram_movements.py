import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import normalize
from src.utils import load_movements_price_dataset

# Carregar o dataset
movements_df = load_movements_price_dataset()
movements = movements_df.drop(['company'], axis=1)
companies = movements_df['company'].values

# Normalizar os dados: normalized_movements
normalized_movements = normalize(movements)

# Calcular o linkage: mergings
mergings = linkage(normalized_movements, method='complete')

# Plotar o dendrograma
dendrogram(mergings,
           labels=companies,
           leaf_rotation=90,
           leaf_font_size=10)
plt.show()
