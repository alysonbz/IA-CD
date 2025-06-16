import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import numpy as np

# Carregar dataset ajustado
df = pd.read_csv("regressao_ajustado.csv")

# Separar preditores (X) e alvo (y)
X = df.drop(columns=["G3"])
y = df["G3"]

# Modelos a serem testados
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1)
}

# Funções para calcular RMSE (como negativo para usar no cross_val_score)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)

# Avaliação com cross_val_score
results = []
for name, model in models.items():
    neg_rmse_scores = cross_val_score(model, X, y, cv=5, scoring=rmse_scorer)
    r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

    results.append({
        "Modelo": name,
        "RMSE médio": -np.mean(neg_rmse_scores),
        "R² médio": np.mean(r2_scores)
    })

# Exibir resultados em tabela
results_df = pd.DataFrame(results)
print(results_df.sort_values(by="R² médio", ascending=False))