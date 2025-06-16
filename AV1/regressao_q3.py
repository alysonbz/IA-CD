import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np

# Carrega os dados
df = pd.read_csv("regressao_ajustado.csv")
X = df.drop(columns=['Weight'])
y = df['Weight']

# Função de avaliação: RMSE negativo
rmse_scorer = make_scorer(mean_squared_error, squared=False)

# Modelos
modelos = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1)
}

for nome, modelo in modelos.items():
    rmse = -cross_val_score(modelo, X, y, scoring=rmse_scorer, cv=5).mean()
    r2 = cross_val_score(modelo, X, y, scoring='r2', cv=5).mean()
    print(f"{nome}: RMSE = {rmse:.2f}, R² = {r2:.4f}")
