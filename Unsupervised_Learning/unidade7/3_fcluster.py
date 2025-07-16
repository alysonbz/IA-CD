import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from src.utils import load_movements_price_dataset
from sklearn.preprocessing import normalize

# Carregar os dados
movements_df = load_movements_price_dataset()
movements = movements_df.drop(['company'], axis=1)
companies = movements_df['company'].values

# Normalizar os dados
normalized_movements = normalize(movements)

# Calcular o linkage
mergings = linkage(normalized_movements, method='complete')

# Usar fcluster para extrair os rótulos dos clusters (ex: 5 clusters)
labels = fcluster(mergings, t=5, criterion='maxclust')

# Criar um DataFrame com os rótulos e os nomes das empresas
df = pd.DataFrame({'label': labels, 'company': companies})

# Criar uma tabela cruzada
ct = pd.crosstab(df['label'], df['company'])

# Exibir a tabela
print(ct)
