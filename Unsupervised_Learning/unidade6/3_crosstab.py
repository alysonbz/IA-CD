import pandas as pd
from src.utils import load_grains_dataset
from sklearn.cluster import KMeans

# Carrega o dataset
samples_df = load_grains_dataset()
samples = samples_df.drop(['variety', 'variety_number'], axis=1)
varieties = samples_df['variety'].values

# Cria o modelo KMeans com 3 clusters
model = KMeans(n_clusters=3, random_state=42)

# Ajusta o modelo e obtém os rótulos dos clusters
labels = model.fit_predict(samples)

# Cria um DataFrame com os rótulos e as variedades verdadeiras
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Cria a tabela cruzada entre os rótulos dos clusters e as variedades reais
ct = pd.crosstab(df['labels'], df['varieties'])

# Exibe a tabela cruzada
print(ct)