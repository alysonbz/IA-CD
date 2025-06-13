import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def print_results_table(headers, data):
    # Inicializar a largura de cada coluna com o comprimento do cabeçalho
    col_widths = [len(str(header)) for header in headers]

    # Atualizar com o comprimento máximo dos dados
    for row in data:
        for i, item in enumerate(row):
            if i < len(col_widths):  # Proteção extra
                col_widths[i] = max(col_widths[i], len(str(item)))
            else:
                col_widths.append(len(str(item)))

    # Imprimir cabeçalho
    header_row = " | ".join([str(headers[i]).ljust(col_widths[i]) for i in range(len(headers))])
    print("-" * len(header_row))
    print(header_row)
    print("-" * len(header_row))

    # Imprimir dados
    for row in data:
        print(" | ".join([str(row[i]).ljust(col_widths[i]) for i in range(len(row))]))
    print("-" * len(header_row))


# Início da execução
print("Executando comparação de modelos de regressão...")

# Carregar dados
df = pd.read_csv('housing.csv')
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Padronizar features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Criar modelos
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1)
}

# Métricas para avaliação
scorers = {
    'RMSE': make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred))),
    'R2': make_scorer(r2_score)
}

# Avaliar modelos com cross-validation
results = []
for name, model in models.items():
    cv_rmse = -cross_val_score(model, X_scaled, y, cv=5, scoring=scorers['RMSE']).mean()
    cv_r2 = cross_val_score(model, X_scaled, y, cv=5, scoring=scorers['R2']).mean()
    results.append([name, f"{cv_rmse:.2f}", f"{cv_r2:.4f}"])

# Exibir resultados em tabela
headers = ["Modelo", "RMSE (CV)", "R² (CV)"]
print("\nComparação dos Modelos:")
print_results_table(headers, results)

