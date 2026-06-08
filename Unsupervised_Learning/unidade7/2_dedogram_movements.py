import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import normalize
from src.utils import load_movements_price_dataset

movements_df = load_movements_price_dataset()
movements = movements_df.drop(['company'],axis=1)
companies = movements_df['company'].values

# Normalize the movements: normalized_movements
normalized_movements = normalize(movements)

# Calculate the linkage: mergings
mergings = linkage(normalized_movements, method='complete')

# Plot the dendrogram
plt.figure(figsize=(10, 5))
dendrogram(
    mergings,
    labels=companies,
    leaf_rotation=90,
    leaf_font_size=6,
)
plt.show()
