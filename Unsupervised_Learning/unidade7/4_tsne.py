# Import TSNE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.utils import load_grains_dataset

samples_df = load_grains_dataset()
samples = samples_df.drop(['variety','variety_number'], axis=1)
variety_numbers = samples_df['variety_number'].values

# Create a TSNE instance
model = TSNE()

# Apply fit_transform to samples
tsne_features = model.fit_transform(samples)

# Select the 0th feature
xs = tsne_features[:, 0]

# Select the 1st feature
ys = tsne_features[:, 1]

# Scatter plot, coloring by variety_numbers
plt.scatter(xs, ys, c=variety_numbers)

plt.show()