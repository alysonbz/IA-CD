import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def carregar_dataset(caminho: str) -> pd.DataFrame:

    df = pd.read_csv(caminho)

    print("=" * 55)
    print("EXPLORAÇÃO INICIAL DO DATASET")
    print("=" * 55)
    print(f"Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas\n")
    print("Tipos de dados:")
    print(df.dtypes)
    print("\nValores ausentes por coluna:")
    print(df.isnull().sum())
    print("\nEstatísticas descritivas:")
    print(df.describe().round(2))

    return df


def tratar_dataset(df: pd.DataFrame) -> pd.DataFrame:

    colunas_remover = ['date', 'rv1', 'rv2']
    df = df.drop(columns=[c for c in colunas_remover if c in df.columns])
    n_antes = len(df)
    df = df.dropna()
    n_removidos = n_antes - len(df)

    print(f"\nColunas removidas: {colunas_remover}")
    print(f"Linhas removidas (nulos): {n_removidos}")
    print(f"Dataset final: {df.shape[0]} linhas x {df.shape[1]} colunas")

    return df

def analisar_atributos(df: pd.DataFrame, alvo: str = 'Appliances') -> pd.Series:

    corr_alvo = df.corr()[[alvo]].drop(alvo).sort_values(alvo, ascending=False)

    print("\n" + "=" * 55)
    print("CORRELAÇÃO COM A VARIÁVEL-ALVO (Appliances)")
    print("=" * 55)
    print(corr_alvo.round(4))

    plt.figure(figsize=(14, 10))
    sns.heatmap(df.corr(), cmap='coolwarm', center=0, annot=False,
                linewidths=0.3, linecolor='white')
    plt.title('Mapa de Correlação entre Todos os Atributos', fontsize=13)
    plt.tight_layout()
    plt.savefig('heatmap_correlacao.png', dpi=120)
    plt.close()
    print("\nHeatmap salvo em: heatmap_correlacao.png")

    plt.figure(figsize=(10, 6))
    corr_alvo[alvo].plot(kind='barh', color='steelblue', edgecolor='white')
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title('Correlação de Pearson de cada Atributo com Appliances', fontsize=12)
    plt.xlabel('Correlação')
    plt.tight_layout()
    plt.savefig('correlacao_atributos.png', dpi=120)
    plt.close()
    print("Gráfico de correlação salvo em: correlacao_atributos.png")

    atributo_top = corr_alvo[alvo].abs().idxmax()
    print(f"\nAtributo mais correlacionado: '{atributo_top}' "
          f"(r = {corr_alvo.loc[atributo_top, alvo]:.4f})")

    return corr_alvo[alvo]

def preparar_dados(df: pd.DataFrame, alvo: str = 'Appliances'):

    X = df.drop(columns=[alvo])
    y = df[alvo]
    feature_names = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print("\n" + "=" * 55)
    print("DIVISÃO TREINO / TESTE")
    print("=" * 55)
    print(f"Treino: {X_train.shape[0]} amostras")
    print(f"Teste:  {X_test.shape[0]} amostras")

    return X_train, X_test, y_train, y_test, scaler, feature_names

def treinar_modelos(X_train, X_test, y_train, y_test) -> tuple:

    modelos = {
        'Linear': LinearRegression(),
        'Ridge':  Ridge(alpha=1.0),
        'Lasso':  Lasso(alpha=0.1, max_iter=10000),
    }

    metricas = {}

    print("\n" + "=" * 55)
    print("TREINO E AVALIAÇÃO (split treino/teste)")
    print("=" * 55)

    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2  = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)

        metricas[nome] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

        print(f"\n{nome}:")
        print(f"  MSE  = {mse:.4f}")
        print(f"  RMSE = {rmse:.4f}")
        print(f"  MAE  = {mae:.4f}")
        print(f"  R²   = {r2:.4f}")

    return modelos, metricas

def regressao_simples_e_plot(df: pd.DataFrame,
                              atributo: str,
                              alvo: str = 'Appliances'):

    X_s = df[[atributo]].values
    y_s = df[alvo].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_s, y_s, test_size=0.2, random_state=42
    )

    modelo_simples = LinearRegression()
    modelo_simples.fit(X_tr, y_tr)
    y_pred_te = modelo_simples.predict(X_te)

    r2_s = r2_score(y_te, y_pred_te)
    print("\n" + "=" * 55)
    print(f"REGRESSÃO SIMPLES — atributo: '{atributo}'")
    print("=" * 55)
    print(f"Coeficiente: {modelo_simples.coef_[0]:.4f}")
    print(f"Intercepto:  {modelo_simples.intercept_:.4f}")
    print(f"R²:          {r2_s:.4f}")

    x_line = np.linspace(X_s.min(), X_s.max(), 300).reshape(-1, 1)
    y_line = modelo_simples.predict(x_line)

    plt.figure(figsize=(8, 5))
    plt.scatter(X_te, y_te, alpha=0.25, s=12, color='steelblue',
                label='Dados de teste')
    plt.plot(x_line, y_line, color='firebrick', linewidth=2,
             label=f'Reta ajustada (R²={r2_s:.3f})')
    plt.xlabel(atributo)
    plt.ylabel(alvo)
    plt.title(f'Regressão Linear Simples: {atributo} → {alvo}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reta_regressao_simples.png', dpi=120)
    plt.close()
    print("Gráfico salvo em: reta_regressao_simples.png")

def validacao_cruzada(modelos: dict, X, y, k: int = 5) -> dict:

    print("\n" + "=" * 55)
    print(f"VALIDAÇÃO CRUZADA ({k}-fold)")
    print("=" * 55)

    cv_resultados = {}
    for nome, modelo in modelos.items():
        scores = cross_val_score(modelo, X, y, cv=k, scoring='r2')
        cv_resultados[nome] = {
            'media_R2': scores.mean(),
            'std_R2':   scores.std(),
            'scores':   scores,
        }
        print(f"{nome:10s}: R² = {scores.mean():.4f} ± {scores.std():.4f}")

    return cv_resultados

def grid_search_regularizacao(X_train, y_train) -> dict:

    grade_alphas = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0]}
    melhores = {}

    print("\n" + "=" * 55)
    print("GRID SEARCH — Ridge e Lasso")
    print("=" * 55)

    for nome, classe in [('Ridge', Ridge), ('Lasso', Lasso)]:
        gs = GridSearchCV(
            estimator=classe(max_iter=10000),
            param_grid=grade_alphas,
            cv=5,
            scoring='r2',
            n_jobs=-1,
        )
        gs.fit(X_train, y_train)
        melhores[nome] = {
            'modelo': gs.best_estimator_,
            'alpha':  gs.best_params_['alpha'],
            'R2_cv':  gs.best_score_,
        }
        print(f"{nome:10s}: melhor alpha = {gs.best_params_['alpha']}"
              f"  |  R² (CV) = {gs.best_score_:.4f}")

    return melhores

def comparar_resultados(metricas_split: dict,
                        cv_resultados: dict,
                        gs_melhores: dict,
                        X_test,
                        y_test):

    print("\n" + "=" * 65)
    print("COMPARATIVO GERAL DOS MODELOS")
    print("=" * 65)
    print(f"{'Modelo':<12} {'R² split':>10} {'R² CV ± std':>18} {'R² GS':>10}")
    print("-" * 65)

    for nome in ['Linear', 'Ridge', 'Lasso']:
        r2_split = metricas_split[nome]['R2']
        media_cv = cv_resultados[nome]['media_R2']
        std_cv   = cv_resultados[nome]['std_R2']

        if nome in gs_melhores:
            y_gs = gs_melhores[nome]['modelo'].predict(X_test)
            r2_gs = f"{r2_score(y_test, y_gs):.4f}"
        else:
            r2_gs = "N/A"

        print(f"{nome:<12} {r2_split:>10.4f} "
              f"{media_cv:>10.4f} ± {std_cv:.4f}   {r2_gs:>10}")


def main():

    CAMINHO = 'energydata_complete.csv'
    ALVO = 'Appliances'

    df_bruto = carregar_dataset(CAMINHO)
    df = tratar_dataset(df_bruto)

    corr = analisar_atributos(df, alvo=ALVO)
    atributo_top = corr.abs().idxmax()

    X_train, X_test, y_train, y_test, scaler, features = preparar_dados(df, ALVO)

    from sklearn.preprocessing import StandardScaler as _SS
    X_full = _SS().fit_transform(df.drop(columns=[ALVO]))
    y_full = df[ALVO]

    modelos, metricas = treinar_modelos(X_train, X_test, y_train, y_test)

    regressao_simples_e_plot(df, atributo=atributo_top, alvo=ALVO)

    cv_res = validacao_cruzada(modelos, X_full, y_full, k=5)

    gs_melhores = grid_search_regularizacao(X_train, y_train)

    comparar_resultados(metricas, cv_res, gs_melhores, X_test, y_test)


if __name__ == '__main__':
    main()