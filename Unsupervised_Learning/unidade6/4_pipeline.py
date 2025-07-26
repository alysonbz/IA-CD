import pandas as pd

# Perform the necessary imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

from src.utils import load_fish_dataset

# Carregar os dados
samples_df = load_fish_dataset()
samples = samples_df.drop(['specie'], axis=1)
species = samples_df['specie'].values

# Criar o scaler
scaler = StandardScaler()

# Criar o modelo KMeans com 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)

# Criar o pipeline com scaler + kmeans
pipeline = make_pipeline(scaler, kmeans)

# Ajustar o pipeline aos dados
pipeline.fit(samples)

# Obter os rótulos dos clusters
labels = pipeline.predict(samples)

# Criar um DataFrame com os rótulos e as espécies reais
df = pd.DataFrame({'labels': labels, 'species': species})

# Criar a tabela cruzada
ct = pd.crosstab(df['labels'], df['species'])

# Mostrar a tabela
print(ct)