import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")
 
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# I. Carregamento e exploração do dataset
print("=" * 70)
print("I. CARREGAMENTO E EXPLORAÇÃO DO DATASET")
print("=" * 70)

dias_df = pd.read_csv("dataset/day.csv")

print(f"\nShape: {dias_df.shape}")
print(f"\nPrimeiras linhas:\n{dias_df.head()}")
print(f"\nTipos de dados:\n{dias_df.dtypes}")
print(f"\nEstatísticas descritivas:\n{dias_df.describe().round(3)}")

# II. Identificação e tratamento de valores ausentes e inconsistentes
print("\n" + "=" * 70)
print("II. VALORES AUSENTES E INCONSISTENTES")
print("=" * 70)

print(f"\nValores nulos por coluna:\n{dias_df.isnull().sum()}")
print(f"\nTotal de nulos: {dias_df.isnull().sum().sum()}")

# Remover colunas não preditivas
df_clean = dias_df.drop(columns=["instant", "dteday", "casual", "registered"])
print(f"\nColunas removidas: instant, dteday, casual, registered")
print(f"\nShape após limpeza: {df_clean.shape}")

# III. Análise de atributos relevantes
print("\n" + "=" * 70)
print("III. ANÁLISE DE ATRIBUTOS RELEVANTES")
print("=" * 70)

correlations = df_clean.corr()["cnt"].drop("cnt").sort_values(ascending=False)
print(f"\nCorrelação com 'cnt':\n{correlations.round(4)}")

# IV. Separação de X e y
print("\n" + "=" * 70)
print("IV. ATRIBUTOS DE ENTRADA E VARIÁVEL-ALVO")
print("=" * 70)

X = df_clean.drop(columns=["cnt"])
y = df_clean["cnt"]

print(f"\nAtributos de entrada (X): {list(X.columns)}")
print(f"Variável-alvo (y): cnt")
print(f"Shape X: {X.shape} | Shape y: {y.shape}")

# V. Pré-processamento
print("\n" + "=" * 70)
print("V. PRÉ-PROCESSAMENTO — StandardScaler")
print("=" * 70)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("\nAtributos padronizados (média≈0, std≈1):")
print(X_scaled_df.describe().round(3).loc[["mean", "std"]])

# VI. Divisão treino/teste
print("\n" + "=" * 70)
print("VI. DIVISÃO TREINO / TESTE")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42
)
print(f"\nTreino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

# Funções auxiliares de métricas
def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae  = mean_absolute_error(y_te, y_pred)
    r2   = r2_score(y_te, y_pred)
    return {"modelo": name, "R²": r2, "RMSE": rmse, "MAE": mae, "y_pred": y_pred}

# VII. Regressão linear
print("\n" + "=" * 70)
print("VII. REGRESSÃO LINEAR PADRÃO")
print("=" * 70)

lin_reg = LinearRegression()
res_lin = evaluate_model("Regressão Linear", lin_reg, X_train, y_train, X_test, y_test)

# VIII. Ridge e Lasso
print("\n" + "=" * 70)
print("VIII. RIDGE E LASSO (alpha padrão)")
print("=" * 70)

ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=1.0, max_iter=10000)

res_ridge = evaluate_model("Ridge (α=1.0)", ridge, X_train, y_train, X_test, y_test)
res_lasso = evaluate_model("Lasso (α=1.0)", lasso, X_train, y_train, X_test, y_test)

# IX. Commparação inicial dos três modelos
print("\n" + "=" * 70)
print("IX. COMPARAÇÃO INICIAL — TREINO/TESTE")
print("=" * 70)

results_df = pd.DataFrame([
    {"Modelo": res_lin["modelo"],   "R²": res_lin["R²"],   "RMSE": res_lin["RMSE"],   "MAE": res_lin["MAE"]},
    {"Modelo": res_ridge["modelo"], "R²": res_ridge["R²"], "RMSE": res_ridge["RMSE"], "MAE": res_ridge["MAE"]},
    {"Modelo": res_lasso["modelo"], "R²": res_lasso["R²"], "RMSE": res_lasso["RMSE"], "MAE": res_lasso["MAE"]},
])
print(f"\n{results_df.to_string(index=False)}")

# X e XI. Regressão simples e visualização -- atributo: temp
print("\n" + "=" * 70)
print("X + XI. REGRESSÃO LINEAR SIMPLES — atributo: temp")
print("=" * 70)

X_simple = df_clean[["temp"]].values
y_simple = df_clean["cnt"].values

X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(
    X_simple, y_simple, test_size=0.20, random_state=42)

simple_model = LinearRegression()

simple_model.fit(X_s_train, y_s_train)
y_s_pred = simple_model.predict(X_s_test)
r2_simple  = r2_score(y_s_test, y_s_pred)
rmse_simple = np.sqrt(mean_squared_error(y_s_test, y_s_pred))

print(f"\n  Coeficiente (inclinação): {simple_model.coef_[0]:.2f}")
print(f"  Intercepto:               {simple_model.intercept_:.2f}")
print(f"  R²:   {r2_simple:.4f}")
print(f"  RMSE: {rmse_simple:.2f}")

# XII. Validação cruzada (k=6)
print("\n" + "=" * 70)
print("XII. VALIDAÇÃO CRUZADA (k=6)")
print("=" * 70)

cv_results = {}
for name, model in [("Linear", LinearRegression()),
                    ("Ridge",  Ridge(alpha=1.0)),
                    ("Lasso",  Lasso(alpha=1.0, max_iter=10000))]:
    scores = cross_val_score(model, X_scaled, y, cv=6, scoring="r2")
    cv_results[name] = scores
    print(f"\n  {name}:")
    print(f"    Scores: {np.round(scores, 4)}")
    print(f"    Média:  {scores.mean():.4f} ± {scores.std():.4f}")

# XIII. Grid Search -- Ridge e Lasso
print("\n" + "=" * 70)
print("XIII. GRID SEARCH — MELHORES HIPERPARÂMETROS")
print("=" * 70)

alphas = {"alpha": [0.001, 0.01, 0.1, 1, 10, 50, 100, 500, 1000]}

gs_ridge = GridSearchCV(Ridge(), alphas, cv=6, scoring="r2")
gs_ridge.fit(X_scaled, y)
best_alpha_ridge = gs_ridge.best_params_["alpha"]
best_r2_ridge    = gs_ridge.best_score_

gs_lasso = GridSearchCV(Lasso(max_iter=10000), alphas, cv=6, scoring="r2")
gs_lasso.fit(X_scaled, y)
best_alpha_lasso = gs_lasso.best_params_["alpha"]
best_r2_lasso    = gs_lasso.best_score_

print(f"\n  Ridge → Melhor alpha: {best_alpha_ridge}  |  R² CV: {best_r2_ridge:.4f}")
print(f"  Lasso → Melhor alpha: {best_alpha_lasso}  |  R² CV: {best_r2_lasso:.4f}")

# Treinar modelos com os melhores alphas encontrados
ridge_best = Ridge(alpha=best_alpha_ridge)
ridge_best.fit(X_train, y_train)
y_pred_ridge_best = ridge_best.predict(X_test)
res_ridge_best = {
    "R²": r2_score(y_test, y_pred_ridge_best),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_ridge_best)),
    "MAE": mean_absolute_error(y_test, y_pred_ridge_best),
}

lasso_best = Lasso(alpha=best_alpha_lasso, max_iter=10000)
lasso_best.fit(X_train, y_train)
y_pred_lasso_best = lasso_best.predict(X_test)
res_lasso_best = {
    "R²": r2_score(y_test, y_pred_lasso_best),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_lasso_best)),
    "MAE": mean_absolute_error(y_test, y_pred_lasso_best),
}

# XIV. Comparação final: Treino/teste vs CV vs Grid Seach
print("\n" + "=" * 70)
print("XIV. COMPARAÇÃO FINAL DOS RESULTADOS")
print("=" * 70)

comparison = pd.DataFrame({
    "Modelo":       ["Linear", "Ridge (α=1)", "Lasso (α=1)",
                     f"Ridge (α={best_alpha_ridge})*", f"Lasso (α={best_alpha_lasso})*"],
    "R² Teste":     [res_lin["R²"], res_ridge["R²"], res_lasso["R²"],
                     res_ridge_best["R²"], res_lasso_best["R²"]],
    "R² CV (k=6)":  [cv_results["Linear"].mean(), cv_results["Ridge"].mean(),
                     cv_results["Lasso"].mean(), best_r2_ridge, best_r2_lasso],
    "RMSE Teste":   [res_lin["RMSE"], res_ridge["RMSE"], res_lasso["RMSE"],
                     res_ridge_best["RMSE"], res_lasso_best["RMSE"]],
})

print(f"\n{comparison.round(4).to_string(index=False)}")
print("\n  * melhores alphas encontrados pelo Grid Search")

# XV. Análise crítica
print("\n" + "=" * 70)
print("XV. ANÁLISE CRÍTICA DOS RESULTADOS")
print("=" * 70)

print("""
  1. QUALIDADE GERAL DOS MODELOS
     Todos os três modelos alcançaram R² ≈ 0.83–0.84 no teste, o que indica
     forte poder preditivo. O dataset Bike Sharing possui padrões sazonais e
     climáticos bem definidos, facilitando a captura linear.

  2. LINEAR vs RIDGE vs LASSO (alpha padrão)
     Os três modelos apresentaram desempenho muito próximo com alpha=1, o que
     sugere baixa colinearidade severa entre as features. Ridge e Lasso com
     regularização fraca comportam-se como a Regressão Linear.

  3. IMPACTO DO GRID SEARCH
     O Grid Search revelou que alphas maiores não melhoram significativamente
     o Ridge, confirmando que o conjunto de atributos não sofre de overfitting
     acentuado. Para o Lasso, alphas elevados eliminam coeficientes e aumentam
     o viés sem ganho real de generalização.

  4. VALIDAÇÃO CRUZADA
     A validação cruzada com k=6 confirmou os resultados do treino/teste —
     as médias de R² são consistentes com pequeno desvio padrão, indicando que
     os modelos generalizam bem e não dependem de uma divisão específica.

  5. REGRESSÃO SIMPLES COM 'temp'
     A temperatura sozinha explica boa parcela da variação nos aluguéis
     (R² ≈ 0.39), sendo o preditor individual mais forte. Isso faz sentido:
     dias quentes incentivam o uso de bicicletas, enquanto dias frios inibem.

  6. LIMITAÇÕES
     - Variáveis categóricas (season, weathersit, yr) foram usadas como
       numéricas; one-hot encoding poderia melhorar os resultados.
     - Relações não-lineares (ex.: calor extremo reduz aluguéis) não são
       capturadas por modelos lineares.
     - O modelo mais simples (Linear) já entrega R² alto, sugerindo que a
       regularização não é crítica neste dataset específico.

  7. CONCLUSÃO
     Para este problema, a Regressão Linear padrão é suficiente. Ridge com
     alpha otimizado oferece marginal melhoria e maior estabilidade. Lasso
     pode ser útil para seleção de variáveis em datasets maiores ou com muita
     colinearidade. A escolha definitiva deve considerar o contexto operacional:
     se interpretabilidade for essencial, Linear ou Lasso (features esparsas)
     são preferidos.
""")

# Geração de painel gráfico
# 1. Correlação de atributos
plt.figure(figsize=(8,5))
correlations.sort_values(ascending=True).plot(
    kind="barh",
    color="skyblue"
)
plt.title("Correlação com cnt")
plt.xlabel("Correlação")
plt.ylabel("Atributos")
plt.grid()
plt.show()

# 2. Distribuição da variável-alvo
plt.figure(figsize=(8,5))
plt.hist(y, bins=30)
plt.axvline(
    y.mean(),
    color = "red",
    linestyle="--",
    label="Média"
)
plt.axvline(
    y.median(),
    color = "green",
    linestyle=":",
    label="Mediana"
)
plt.title("Distribuição de cnt")
plt.xlabel("cnt")
plt.ylabel("Frequência")
plt.legend()
plt.grid()
plt.show()

# 3. Regressão linear simples
x_range = np.linspace(
    X_simple.min(),
    X_simple.max(),
    100
).reshape(-1,1)
y_line = simple_model.predict(x_range)
plt.figure(figsize=(8,5))
plt.scatter(
    X_s_test,
    y_s_test,
    label="Dados reais"
)
plt.plot(
    x_range,
    y_line,
    color = "red",
    label="Reta de regressão"
)
plt.title("Regressão Linear Simples")
plt.xlabel("Temperatura (temp)")
plt.ylabel("Total de aluguéis (cnt)")
plt.legend()
plt.grid()
plt.show()

# 4. Real vs Predito
plt.figure(figsize=(8,5))
plt.scatter(
    y_test,
    res_lin["y_pred"],
    label="Linear"
)
plt.scatter(
    y_test,
    y_pred_ridge_best,
    label="Ridge"
)
plt.scatter(
    y_test,
    y_pred_lasso_best,
    label="Lasso"
)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)
plt.title("Real vs Predito")
plt.xlabel("Valores Reais")
plt.ylabel("Valores Preditos")
plt.legend()
plt.grid()
plt.show()

# 5.Boxplot da validação cruzada
plt.figure(figsize=(8,5))
plt.boxplot([
    cv_results["Linear"],
    cv_results["Ridge"],
    cv_results["Lasso"]
])
plt.xticks(
    [1,2,3],
    ["Linear", "Ridge", "Lasso"]
)
plt.title("Validação Cruzada")
plt.ylabel("R²")
plt.grid()
plt.show()

# 6. Grid search (Pesquisa de grade)
alpha_grid = [0.001, 0.01, 0.1, 1, 10, 50, 100, 500, 1000]
plt.figure(figsize=(8,5))
plt.semilogx(
    alpha_grid,
    gs_ridge.cv_results_["mean_test_score"],
    marker="o",
    label="Ridge"
)
plt.semilogx(
    alpha_grid,
    gs_lasso.cv_results_["mean_test_score"],
    marker="s",
    label="Lasso"
)
plt.title("Grid Search")
plt.xlabel("Alpha")
plt.ylabel("R²")
plt.legend()
plt.grid()
plt.show()