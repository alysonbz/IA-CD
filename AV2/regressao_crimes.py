import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------
# 1. Carregando e explorando o dataset
# -------------------------------------------------------

def carregar_dados(caminho):
    """
    Carrega o CSV e faz uma exploração inicial completa do dataset.
    Mostra shape, tipos de dados, distribuição da variável alvo
    e quantidade de valores ausentes por coluna.
    """
    df = pd.read_csv(caminho)

    print("--- Exploração inicial ---")
    print(f"Linhas: {df.shape[0]}, Colunas: {df.shape[1]}")

    print("\nTipos de dados:")
    print(df.dtypes.value_counts())

    print("\nPrimeiras linhas:")
    print(df.head(3))

    print("\nEstatísticas gerais (colunas numéricas):")
    print(df.describe().T[["mean", "std", "min", "max"]].round(3).to_string())

    print("\nEstatísticas da variável alvo (ViolentCrimesPerPop):")
    print(df["ViolentCrimesPerPop"].describe())

    print("\nValores ausentes por coluna (apenas as que têm):")
    missing = df.isnull().sum()
    print(missing[missing > 0].sort_values(ascending=False))


    return df


# -------------------------------------------------------
# 2. Pré-processamento
# -------------------------------------------------------

def preprocessar(df):
    """
    Trata valores ausentes e remove colunas desnecessárias.
    As colunas do tipo 'Lemas' têm quase 84% de missing, então
    não faz sentido tentar imputar - melhor remover mesmo.
    Colunas identificadoras (nome, estado etc) também são removidas
    porque não carregam informação útil pra o modelo.
    """

    # removendo colunas que são só identificadores, não servem como features
    cols_remover = ["communityname", "state", "county", "community", "fold"]
    df = df.drop(columns=cols_remover)

    # vendo quais colunas têm muitos valores ausentes
    proporcao_missing = df.isnull().mean()
    muitos_missing = proporcao_missing[proporcao_missing > 0.5].index.tolist()

    print(f"\nColunas removidas por excesso de missing ({len(muitos_missing)}):")
    print(muitos_missing)

    df = df.drop(columns=muitos_missing)

    # o restante dos missings imputa pela mediana
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    print(f"\nMissing values restantes: {df.isnull().sum().sum()}")
    print(f"Shape após pré-processamento: {df.shape}")

    y = df["ViolentCrimesPerPop"]
    X = df.drop(columns=["ViolentCrimesPerPop"])

    return X, y


# -------------------------------------------------------
# 3. Análise dos atributos relevantes
# -------------------------------------------------------

def analisar_atributos(X, y):
    """
    Calcula correlação de Pearson entre cada atributo e o target.
    Atributos com correlação alta (positiva ou negativa) são os mais
    úteis pra o modelo aprender. Usamos o valor absoluto pra ordenar
    independente do sinal.
    """

    corr = X.corrwith(y).abs().sort_values(ascending=False)

    print("\n--- Top 15 atributos mais correlacionados ---")
    print(corr.head(15).round(4))

    # plotando
    fig, ax = plt.subplots(figsize=(9, 6))
    corr.head(15).sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Correlação absoluta com ViolentCrimesPerPop")
    ax.set_title("Atributos mais relevantes para o modelo")
    plt.tight_layout()
    plt.savefig("correlacoes_atributos.png", dpi=150)
    plt.close()
    print("[gráfico salvo: correlacoes_atributos.png]")

    return corr


# -------------------------------------------------------
# 4. Dividindo treino e teste
# -------------------------------------------------------

def dividir(X, y):
    """
    Divide os dados em 80% treino e 20% teste.
    O random_state garante que a divisão seja sempre a mesma,
    o que é importante pra reprodutibilidade dos resultados.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTreino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")
    return X_train, X_test, y_train, y_test


# -------------------------------------------------------
# 5. Normalização
# -------------------------------------------------------

def normalizar(X_train, X_test):
    """
    Aplica Z-score (StandardScaler) nos dados.
    Normalizar é essencial pra Ridge e Lasso porque as penalizações
    deles são sensíveis à escala - sem isso, variáveis com valores
    grandes seriam penalizadas mais do que deveriam.
    O fit é feito só no treino pra não vazar informação do teste.
    """
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return X_train_sc, X_test_sc, scaler


# -------------------------------------------------------
# 6. Função auxiliar de métricas
# -------------------------------------------------------

def mostrar_metricas(nome, y_real, y_pred):
    """
    Calcula e imprime MAE, MSE, RMSE e R².
    Retorna um dicionário com os valores pra usar nas comparações.
    """
    mae  = mean_absolute_error(y_real, y_pred)
    mse  = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_real, y_pred)

    print(f"\n  {nome}")
    print(f"    MAE : {mae:.4f}")
    print(f"    MSE : {mse:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    R²  : {r2:.4f}")

    return {"modelo": nome, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


# -------------------------------------------------------
# 7. Treinamento dos modelos
# -------------------------------------------------------

def treinar_modelos(X_train_sc, X_test_sc, y_train, y_test):
    """
    Treina Regressão Linear, Ridge e Lasso e avalia no conjunto de teste.
    - Linear: sem regularização, pode sofrer com multicolinearidade
    - Ridge (L2): encolhe os coeficientes mas não zera nenhum
    - Lasso (L1): pode zerar coeficientes, fazendo seleção de variáveis
    """

    print("\n--- Resultados no conjunto de teste ---")

    resultados = []

    # regressão linear simples, sem regularização
    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)
    resultados.append(mostrar_metricas("Regressão Linear", y_test, lr.predict(X_test_sc)))

    # ridge - penaliza L2, não zera coeficientes
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_sc, y_train)
    resultados.append(mostrar_metricas("Ridge (alpha=1.0)", y_test, ridge.predict(X_test_sc)))

    # lasso - penaliza L1, pode zerar coeficientes (seleção de variáveis)
    lasso = Lasso(alpha=0.001, max_iter=10000)
    lasso.fit(X_train_sc, y_train)
    n_zeros = np.sum(lasso.coef_ == 0)
    print(f"\n    -> Lasso zerou {n_zeros} de {X_train_sc.shape[1]} coeficientes")
    resultados.append(mostrar_metricas("Lasso (alpha=0.001)", y_test, lasso.predict(X_test_sc)))

    return resultados, lr, ridge, lasso


# -------------------------------------------------------
# 8. Regressão simples com 1 atributo + gráfico da reta
# -------------------------------------------------------

def regressao_simples(X, y, corr, X_train, X_test, y_train, y_test):
    """
    Usa apenas o atributo mais correlacionado pra fazer uma regressão simples.
    Escolhemos o atributo de maior correlação com o target porque ele é
    o que melhor explica linearmente a variação nos crimes - faz sentido
    visualizar a reta justamente com ele.
    """
    atributo = corr.index[0]
    print(f"\n--- Regressão simples com '{atributo}' ---")
    print(f"Correlação com target: {corr.iloc[0]:.4f}")
    print(f"Justificativa: maior correlação absoluta entre todos os atributos.")

    X_tr_s = X_train[[atributo]].values
    X_te_s = X_test[[atributo]].values

    modelo_s = LinearRegression()
    modelo_s.fit(X_tr_s, y_train)
    y_pred_s = modelo_s.predict(X_te_s)

    mostrar_metricas(f"Linear simples [{atributo}]", y_test, y_pred_s)

    # gráfico da reta de regressão
    x_linha = np.linspace(X[atributo].min(), X[atributo].max(), 200).reshape(-1, 1)
    y_linha = modelo_s.predict(x_linha)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X_test[atributo], y_test, alpha=0.4, s=18, color="steelblue", label="Dados reais")
    ax.plot(x_linha, y_linha, color="crimson", lw=2, label="Reta de regressão")
    ax.set_xlabel(atributo)
    ax.set_ylabel("ViolentCrimesPerPop")
    ax.set_title(f"Regressão Simples: {atributo}")
    ax.legend()
    plt.tight_layout()
    plt.savefig("regressao_simples.png", dpi=150)
    plt.close()
    print("[gráfico salvo: regressao_simples.png]")


# -------------------------------------------------------
# 9. Validação cruzada
# -------------------------------------------------------

def validacao_cruzada(X_train_sc, y_train):
    """
    Aplica K-Fold com k=5 nos três modelos.
    Mais confiável do que uma única divisão treino/teste porque
    testa o modelo em diferentes subconjuntos dos dados e retorna
    a média - reduz o efeito da sorte na divisão.
    """
    print("\n--- Validação Cruzada (5-fold) ---")

    modelos = [
        ("Regressão Linear", LinearRegression()),
        ("Ridge (alpha=1.0)", Ridge(alpha=1.0)),
        ("Lasso (alpha=0.001)", Lasso(alpha=0.001, max_iter=10000)),
    ]

    scores_cv = {}
    for nome, modelo in modelos:
        scores = cross_val_score(modelo, X_train_sc, y_train, cv=5, scoring="r2")
        scores_cv[nome] = scores.mean()
        print(f"\n  {nome}")
        print(f"    R² por fold: {scores.round(4)}")
        print(f"    Média: {scores.mean():.4f} | Desvio: {scores.std():.4f}")

    return scores_cv


# -------------------------------------------------------
# 10. Grid Search
# -------------------------------------------------------

def grid_search(X_train_sc, y_train, X_test_sc, y_test):
    """
    Busca automática do melhor alpha para Ridge e Lasso usando CV=5.
    Testa cada valor do grid e escolhe o que der maior R² médio
    na validação cruzada. Depois avalia o melhor modelo no teste.
    """

    print("\n--- Grid Search ---")

    grid = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

    gs_ridge = GridSearchCV(Ridge(), grid, cv=5, scoring="r2")
    gs_ridge.fit(X_train_sc, y_train)
    melhor_r = gs_ridge.best_estimator_
    r2_ridge_gs = r2_score(y_test, melhor_r.predict(X_test_sc))
    print(f"\n  Ridge - melhor alpha: {gs_ridge.best_params_['alpha']}")
    print(f"    R² CV   : {gs_ridge.best_score_:.4f}")
    print(f"    R² teste: {r2_ridge_gs:.4f}")

    gs_lasso = GridSearchCV(Lasso(max_iter=10000), grid, cv=5, scoring="r2")
    gs_lasso.fit(X_train_sc, y_train)
    melhor_l = gs_lasso.best_estimator_
    r2_lasso_gs = r2_score(y_test, melhor_l.predict(X_test_sc))
    print(f"\n  Lasso - melhor alpha: {gs_lasso.best_params_['alpha']}")
    print(f"    R² CV   : {gs_lasso.best_score_:.4f}")
    print(f"    R² teste: {r2_lasso_gs:.4f}")

    return gs_ridge, gs_lasso, r2_ridge_gs, r2_lasso_gs


# -------------------------------------------------------
# 11. Tabela comparativa final (treino/teste vs CV vs Grid Search)
# -------------------------------------------------------

def comparar_abordagens(resultados_teste, scores_cv, gs_ridge, gs_lasso,
                         r2_ridge_gs, r2_lasso_gs):
    """
    Compara o R² obtido nas três abordagens de avaliação:
    - Divisão treino/teste simples
    - Validação cruzada (5-fold)
    - Grid Search com CV

    Isso mostra se os resultados são consistentes entre as abordagens
    e qual modelo generaliza melhor de forma mais confiável.
    """

    print("\n--- Comparação entre abordagens de avaliação (R²) ---")

    nomes = ["Regressão Linear", "Ridge", "Lasso"]

    r2_teste = [r["R2"] for r in resultados_teste]
    r2_cv    = [
        scores_cv["Regressão Linear"],
        scores_cv["Ridge (alpha=1.0)"],
        scores_cv["Lasso (alpha=0.001)"],
    ]
    r2_gs    = [
        None,  # grid search não se aplica à linear
        r2_ridge_gs,
        r2_lasso_gs,
    ]

    print(f"\n  {'Modelo':<20} {'Treino/Teste':>14} {'CV (5-fold)':>13} {'Grid Search':>13}")
    print("  " + "-" * 62)
    for i, nome in enumerate(nomes):
        gs_val = f"{r2_gs[i]:.4f}" if r2_gs[i] is not None else "     -"
        print(f"  {nome:<20} {r2_teste[i]:>14.4f} {r2_cv[i]:>13.4f} {gs_val:>13}")

    # gráfico de barras agrupadas
    x = np.arange(len(nomes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, r2_teste, width, label="Treino/Teste", color="#2196F3", alpha=0.85)
    ax.bar(x,         r2_cv,    width, label="Validação Cruzada", color="#FF9800", alpha=0.85)

    # grid search só pra ridge e lasso
    gs_vals = [r2_ridge_gs, r2_lasso_gs]
    ax.bar(x[1:] + width, gs_vals, width, label="Grid Search", color="#4CAF50", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(nomes)
    ax.set_ylabel("R²")
    ax.set_title("Comparação de R² por abordagem de avaliação")
    ax.legend()
    ax.set_ylim(0, 1)
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig("comparacao_abordagens.png", dpi=150)
    plt.close()
    print("\n[gráfico salvo: comparacao_abordagens.png]")


# -------------------------------------------------------
# 12. Gráfico comparativo dos modelos (métricas)
# -------------------------------------------------------

def grafico_comparativo(resultados):
    """Compara MAE, RMSE e R² dos três modelos em barras."""

    df_r = pd.DataFrame(resultados)
    nomes = [m.split(" ")[0] for m in df_r["modelo"]]
    x = np.arange(len(nomes))

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    metricas = ["MAE", "RMSE", "R2"]
    cores = ["#2196F3", "#FF5722", "#4CAF50"]
    titulos = ["MAE (menor = melhor)", "RMSE (menor = melhor)", "R² (maior = melhor)"]

    for i, (m, cor, titulo) in enumerate(zip(metricas, cores, titulos)):
        axes[i].bar(x, df_r[m], color=cor, alpha=0.8, width=0.5)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(nomes, fontsize=10)
        axes[i].set_title(titulo)
        for j, v in enumerate(df_r[m]):
            axes[i].text(j, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)

    plt.suptitle("Comparação entre os modelos", fontsize=13)
    plt.tight_layout()
    plt.savefig("comparacao_modelos.png", dpi=150)
    plt.close()
    print("\n[gráfico salvo: comparacao_modelos.png]")


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    """Executa o pipeline completo de regressão."""

    caminho = "communities.csv"

    df                                      = carregar_dados(caminho)
    X, y                                    = preprocessar(df)
    corr                                    = analisar_atributos(X, y)
    X_train, X_test, y_train, y_test        = dividir(X, y)
    X_train_sc, X_test_sc, scaler          = normalizar(X_train, X_test)
    resultados, lr, ridge, lasso            = treinar_modelos(X_train_sc, X_test_sc, y_train, y_test)
    regressao_simples(X, y, corr, X_train, X_test, y_train, y_test)
    scores_cv                               = validacao_cruzada(X_train_sc, y_train)
    gs_ridge, gs_lasso, r2_rgs, r2_lgs     = grid_search(X_train_sc, y_train, X_test_sc, y_test)
    comparar_abordagens(resultados, scores_cv, gs_ridge, gs_lasso, r2_rgs, r2_lgs)
    grafico_comparativo(resultados)

    print("\nConcluído!")


if __name__ == "__main__":
    main()