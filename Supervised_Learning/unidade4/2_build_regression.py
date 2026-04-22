from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression
sales_df = load_sales_clean_dataset()

y = sales_df["sales"].values
X = sales_df["radio"].values.reshape(-1, 1)

# Create the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X, y)

# Make predictions
predictions = reg.predict(X)


print("5 primeiras condiçõs preditas", predictions[:5])
print("Valores reais", y[0:5])