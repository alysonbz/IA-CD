
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.utils import load_fish_dataset
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

samples = load_fish_dataset()
species = samples['specie'].values
samples = samples.drop(['specie'],axis=1)
scaler = StandardScaler()
scaled_samples = scaler.fit_transform(samples)



# Create a PCA model with components in adequate number: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)

#vizualize scatter plot with dimension reduced
le = LabelEncoder()
species_encoded = le.fit_transform(species)

plt.scatter(pca_features[:, 0], pca_features[:, 1], c=species_encoded, cmap='viridis')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA - Redução de Dimensionalidade (Peixes)')
plt.show()