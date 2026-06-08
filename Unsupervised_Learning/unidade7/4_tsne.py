# Import TSNE

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.utils import load_grains_dataset

samples_df = load_grains_dataset()
samples = samples_df.drop(['variety','variety_number'],axis=1)
variety_numbers = samples_df['variety_number'].values


# Create a TSNE instance: model
model = TSNE(learning_rate=100)
transformed = model.fit_transform(samples)

# Apply fit_transform to samples: tsne_features
tsne_features = transformed

# Select the 0th feature: xs
xs = transformed[:,0]

# Select the 1st feature: ys
ys = transformed[:,1]

# Scatter plot, coloring by variety_numbers
plt.scatter(xs, ys, c=variety_numbers)

plt.show()
