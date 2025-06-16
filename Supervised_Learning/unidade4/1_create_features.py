import numpy as np
from src.utils import load_sales_clean_dataset

# Carrega o dataset
sales_df = load_sales_clean_dataset()

# Cria X a partir da coluna 'radio'
X = sales_df['radio'].values.reshape(-1, 1)  # reshape para virar matriz (necessário para modelos do sklearn)

# Cria y a partir da coluna 'sales'
y = sales_df['sales'].values  # vetor unidimensional

# Verifica as dimensões de X e y
print(X.shape, y.shape)