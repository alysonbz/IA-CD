
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.utils import load_fish_dataset
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

samples = load_fish_dataset()
x = samples['specie']
samples = samples.drop(['specie'],axis=1)
scaler = StandardScaler()
scaled_samples = scaler.fit_transform(samples)


# Create a PCA model with components in adequate number: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
pc_features=pca.transform(scaled_samples)

# Print the shape of pca_features
print(pc_features.shape)

#vizualize scatter plot with dimension reduced
le=LabelEncoder()
labels=le.fit_transform(x)

plt.figure(figsize=(8, 6))
plt.scatter(pc_features[:, 0], pc_features[:, 1], c=labels, cmap='viridis')
plt.show()