from src.utils import load_sales_clean_dataset

sales_df = load_sales_clean_dataset()

# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Prepara os dados
y = sales_df["sales"].values
X = sales_df["radio"].values.reshape(-1, 1)

# Cria o modelo
reg = LinearRegression()

# Treina o modelo com os dados
reg.fit(X, y)

# Faz previsões
predictions = reg.predict(X)

# Exibe as previsões
print(predictions)