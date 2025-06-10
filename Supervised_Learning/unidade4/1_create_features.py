import numpy as np
from src.utils import load_sales_clean_dataset

sales_df = load_sales_clean_dataset()

# Create X from the radio column's values
X = sales_df['radio']
print(sales_df.columns)
# Create y from the sales column's values
y = sales_df['sales']

# Reshape X
X = X.values.reshape(-1, 1)

# Check the shape of the features and targets
print(X.shape,y.shape)