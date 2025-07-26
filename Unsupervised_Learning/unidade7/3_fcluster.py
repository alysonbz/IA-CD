# Perform the necessary imports
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

# Calcular linkage
mergings = linkage(normalized_movements, method='complete')

# Extrair rótulos dos clusters (definindo 10 clusters, por exemplo)
labels = fcluster(mergings, 10, criterion='maxclust')

# Criar DataFrame com os rótulos e as empresas
df = pd.DataFrame({'labels': labels, 'companies': companies})

# Criar tabela cruzada
ct = pd.crosstab(df['labels'], df['companies'])

# Mostrar resultado
print(ct)