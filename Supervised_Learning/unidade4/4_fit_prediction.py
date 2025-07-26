from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Carregar o dataset de vendas
sales_df = load_sales_clean_dataset()

# Importar mean_squared_error
from sklearn.metrics import mean_squared_error

# Criar arrays X e y
X = sales_df.drop(["sales", "date"], axis=1)
y = sales_df["sales"].values

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instanciar o modelo
reg = LinearRegression()

# Ajustar o modelo aos dados
reg.fit(X_train, y_train)

# Fazer previsões
y_pred = reg.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))

# Calcular R²
r_squared = reg.score(X_test, y_test)

# Calcular RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Exibir métricas
print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))