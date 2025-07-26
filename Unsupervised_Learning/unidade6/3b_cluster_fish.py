import pandas as pd
from src.utils import load_fish_dataset
from sklearn.cluster import KMeans

# Carregar o dataset
samples_df = load_fish_dataset()

# Remover a coluna de rótulo
samples = samples_df.drop(['specie'], axis=1)
specie = samples_df['specie'].values

# Criar instância do KMeans com 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)

# Ajustar o modelo e obter os rótulos dos clusters
labels = kmeans.fit_predict(samples)

# Criar DataFrame com os rótulos e as espécies reais
df = pd.DataFrame({'labels': labels, 'specie': specie})

# Criar a tabela cruzada (crosstab)
ct = pd.crosstab(df['labels'], df['specie'])

# Exibir a tabela cruzada
print(ct)