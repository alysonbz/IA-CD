# Perform the necessary imports
import pandas as pd

from scipy.cluster.hierarchy import fcluster,linkage
from src.utils import load_movements_price_dataset
from sklearn.preprocessing import normalize

movements_df = load_movements_price_dataset()
movements = movements_df.drop(['company'],axis=1)
companies = movements_df['company'].values

normalized_movements = normalize(movements)

# Calculate the linkage: mergings
mergings = linkage(normalized_movements, method='complete')

# Use fcluster to extract labels: labels
labels = fcluster(mergings, 1, criterion='distance')

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'Labels': labels,
                   'Companies': companies})

# Create crosstab: ct
ct = pd.crosstab(df['Labels'],df['Companies'])

# Display ct
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(ct)
