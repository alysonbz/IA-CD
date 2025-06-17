import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

# Carregar dataset
df = pd.read_csv('/root/.cache/kagglehub/datasets/vipullrathod/fish-market/versions/1/Fish.csv')
df = df.drop('Species', axis=1)

X = df.drop('Weight', axis=1).values
y = df['Weight'].values

# Modelos
modelos = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso()
}


# Função para RMSE (negativo, pois cross_val_score maximiza a métrica)
def rmse_scorer(y_true, y_pred):
    return -np.sqrt(mean_squared_error(y_true, y_pred))


rmse_score = make_scorer(rmse_scorer, greater_is_better=True)
r2_score_sklearn = make_scorer(r2_score)

resultados = []

for nome, modelo in modelos.items():
    # RMSE com cross_val_score (negativo)
    scores_rmse = cross_val_score(modelo, X, y, scoring=rmse_score, cv=5)
    # R2 com cross_val_score
    scores_r2 = cross_val_score(modelo, X, y, scoring=r2_score_sklearn, cv=5)

    # Média dos scores
    media_rmse = -scores_rmse.mean()  # inverter o sinal para positivo
    media_r2 = scores_r2.mean()

    resultados.append({
        'Modelo': nome,
        'RMSE': media_rmse,
        'R2': media_r2
    })

# Criar DataFrame para visualizar resultados
df_resultados = pd.DataFrame(resultados)

print(df_resultados)

# Identificar o melhor modelo (menor RMSE, maior R2)
melhor_rmse = df_resultados.loc[df_resultados['RMSE'].idxmin()]
melhor_r2 = df_resultados.loc[df_resultados['R2'].idxmax()]

print(f"\nMelhor modelo pelo RMSE: {melhor_rmse['Modelo']} com RMSE = {melhor_rmse['RMSE']:.4f}")
print(f"Melhor modelo pelo R2: {melhor_r2['Modelo']} com R2 = {melhor_r2['R2']:.4f}")
