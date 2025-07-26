# Import pandas
import pandas as pd
# Import Normalizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from src.utils import load_movements_price_dataset

# Carregar os dados
movements_df = load_movements_price_dataset()
movements = movements_df.drop(['company'], axis=1)
companies = movements_df['company'].values

# Criar um normalizador
normalizer = Normalizer()

# Criar modelo KMeans com 10 clusters
kmeans = KMeans(n_clusters=10, random_state=42)

# Criar pipeline com normalizer e kmeans
pipeline = make_pipeline(normalizer, kmeans)

# Ajustar o pipeline aos dados
pipeline.fit(movements)

# Obter os rótulos dos clusters
labels = pipeline.predict(movements)

# Criar DataFrame com labels e empresas
df = pd.DataFrame({'labels': labels, 'companies': companies})

# Exibir o DataFrame ordenado pelos rótulos dos clusters
print(df.sort_values(by='labels'))