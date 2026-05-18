"""
Projeto de Regressão - AV2 Inteligência Artificial

Dataset: Communities and Crime
Modelos: Regressão Linear, Ridge Regression e Lasso Regression

Este arquivo executa um pipeline completo de regressão:
- leitura do dataset via ucimlrepo;
- tratamento de valores ausentes e inconsistências;
- caracterização demográfica do dataset;
- seleção de features por limiar de correlação;
- treinamento dos modelos Linear, Ridge e Lasso;
- avaliação com MAE, MSE, RMSE e R²;
- visualização da reta de regressão simples;
- validação cruzada;
- Grid Search para Ridge e Lasso;
- comparação final dos resultados.
"""

import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

PLOTS_DIR = Path("plots")
CSV_DIR = Path("AV2/csv")
TARGET = "ViolentCrimesPerPop"

TRANSFORMACOES = {
    "ViolentCrimesPerPop": "log",
    "PctKids2Par": "power",
    "PctIlleg": "log",
    "PctFam2Par": "power",
    "racePctWhite": "power",
    "PctYoungKids2Par": "power",
    "PctTeen2Par": "power",
    "racepctblack": "log",
    "pctWInvInc": "power",
    "pctWPubAsst": "log",
    "FemalePctDiv": "power",
    "TotalPctDiv": "power",
    "PctPersOwnOccup": "power",
    "MalePctDivorce": "power",
    "PctPopUnderPov": "log",
    "PctUnemployed": "log",
}


# -------------------------------------------------------
# 1. Carregamento do dataset
# -------------------------------------------------------

def carregar_dados():
    """
    Carrega o dataset Communities and Crime via ucimlrepo (id=183).
    Combina features e target em um único DataFrame.
    """
    dataset = fetch_ucirepo(id=183)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    return df


# -------------------------------------------------------
# 2. Exploração inicial
# -------------------------------------------------------

def explorar_dados(df):
    """
    Mostra uma visão inicial do dataset, incluindo quantidade de linhas e colunas,
    primeiras linhas, tipos de dados e estatísticas descritivas.
    Essa etapa ajuda a compreender a estrutura dos dados antes da modelagem.
    """
    print("--- Exploração Inicial ---")
    print(f"\nShape inicial do dataset: {df.shape}")

    print("\nPrimeiras linhas:")
    print(df.head())

    print("\nTipos de dados:")
    print(df.dtypes)

    print("\nEstatísticas descritivas:")
    print(df.describe().T.round(4))

    print("\nInformações gerais:")
    df.info()


# -------------------------------------------------------
# 3. Valores ausentes e inconsistências
# -------------------------------------------------------

def verificar_missing(df):
    """
    Verifica a quantidade e o percentual de valores ausentes por coluna.
    No Communities and Crime, algumas colunas possuem muitos valores faltantes,
    principalmente atributos relacionados a dados policiais.
    """
    df = df.copy()
    df.replace("?", np.nan, inplace=True)

    missing = df.isna().sum()
    missing_percent = (missing / len(df)) * 100

    tabela_missing = pd.DataFrame({
        "missing": missing,
        "percentual": missing_percent.round(2)
    }).sort_values("missing", ascending=False)

    print("\n--- Valores Ausentes ---")
    print(tabela_missing[tabela_missing["missing"] > 0].head(30))

    return tabela_missing


def plotar_missing(tabela_missing):
    """
    Plota as 20 colunas com maior percentual de valores ausentes.
    O gráfico auxilia na decisão de remover colunas com muitos dados faltando.
    """
    top_missing = tabela_missing[tabela_missing["missing"] > 0].head(20)

    if top_missing.empty:
        print("Nenhum valor ausente encontrado para plotar.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=top_missing.reset_index(),
        x="percentual",
        y="index",
        color="steelblue",
        ax=ax
    )
    ax.set_title("Top 20 colunas com valores ausentes")
    ax.set_xlabel("Percentual de valores ausentes (%)")
    ax.set_ylabel("Colunas")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "missing_values_regressao.png", dpi=150)
    plt.close()

    print("salvo: missing_values_regressao.png")


def preparar_base(df, limite_missing=0.50):
    """
    Realiza a limpeza inicial do dataset.

    Etapas:
    1. Substitui possíveis símbolos '?' por NaN.
    2. Remove 'fold' imediatamente (artefato metodológico do UCI, não é preditor).
    3. Converte colunas numéricas para numeric (preserva colunas categóricas identificadoras).
    4. Remove colunas com 50% ou mais de valores ausentes.
    5. Aplica filtro de esparsidade nas colunas categóricas remanescentes:
       sparsity = n_distinct / n_total > 0.5 → remove (quasi-ID, encoding inútil).
    6. Retorna colunas categóricas que passaram o filtro para encoding posterior.
    """
    df = df.copy()

    df.replace("?", np.nan, inplace=True)

    # fold é artefato de CV do UCI — remove antes de qualquer análise
    if "fold" in df.columns:
        df.drop(columns=["fold"], inplace=True)

    COLS_ID = ["state", "county", "community", "communityname"]
    cols_id_presentes = [c for c in COLS_ID if c in df.columns]
    n_total = len(df)

    # Conversão numérica apenas nas colunas não-categóricas
    for col in df.columns:
        if col not in cols_id_presentes:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filtro de missing >= 50% (aplica em todas as colunas)
    limite = int(n_total * limite_missing)
    colunas_muito_missing = df.columns[df.isna().sum() >= limite].tolist()

    if TARGET in colunas_muito_missing:
        colunas_muito_missing.remove(TARGET)

    categoricas_removidas_missing = [c for c in cols_id_presentes if c in colunas_muito_missing]
    df.drop(columns=colunas_muito_missing, inplace=True)

    # Atualiza lista de categóricas sobreviventes ao filtro de missing
    cols_id_sobreviventes = [c for c in cols_id_presentes if c in df.columns]

    # Filtro de esparsidade apenas nas categóricas que sobreviveram ao missing
    removidas_sparsity = []
    cols_categoricas_mantidas = []

    print("\n--- Filtro de esparsidade nas colunas categóricas (n_distinct / n_total) ---")
    for col in cols_id_sobreviventes:
        n_distinct = df[col].nunique(dropna=True)
        sparsity = n_distinct / n_total
        if sparsity > 0.5:
            removidas_sparsity.append(col)
            status = "REMOVIDA"
        else:
            cols_categoricas_mantidas.append(col)
            status = "MANTIDA "
        print(f"  [{status}] {col}: {n_distinct} distintos | sparsity={sparsity:.4f}")

    print(f"\n--- Pré-processamento inicial ---")
    print(f"Coluna 'fold' removida (artefato metodológico).")
    print(f"Colunas removidas por excesso de missing (>= 50%): {len(colunas_muito_missing)}")
    print(f"\nColunas categóricas removidas por missing:           {categoricas_removidas_missing or 'nenhuma'}")
    print(f"Colunas categóricas removidas por sparsity (> 0.5): {removidas_sparsity or 'nenhuma'}")
    print(f"Colunas categóricas mantidas para encoding:         {cols_categoricas_mantidas or 'nenhuma'}")

    df.drop(columns=removidas_sparsity, inplace=True)

    print(f"\nShape após limpeza inicial: {df.shape}")

    return df, colunas_muito_missing, cols_categoricas_mantidas


# -------------------------------------------------------
# Encoding de colunas categóricas
# -------------------------------------------------------

def codificar_categoricas(df, cols_categoricas):
    """
    Aplica One-Hot Encoding nas colunas categóricas que passaram o filtro de sparsity.
    Usa pd.get_dummies com drop_first=False para manter todas as categorias visíveis
    na análise de correlação.
    Retorna o DataFrame com as colunas originais substituídas pelas dummies.
    """
    if not cols_categoricas:
        return df

    df = pd.get_dummies(df, columns=cols_categoricas, drop_first=False, dtype=float)

    novas = [c for c in df.columns if any(c.startswith(f"{cat}_") for cat in cols_categoricas)]
    print(f"\nOHE aplicado em {cols_categoricas}: {len(novas)} colunas geradas.")

    return df


# -------------------------------------------------------
# Utilitário: base para análise exploratória
# -------------------------------------------------------

def obter_base_para_analise(df):
    """
    Cria uma cópia do dataset com valores ausentes preenchidos pela mediana.
    Essa base é usada apenas para análise exploratória e correlação.
    Para o treinamento dos modelos, a imputação acontece dentro do Pipeline.
    """
    df_analise = df.copy()
    medianas = df_analise.median(numeric_only=True)
    df_analise = df_analise.fillna(medianas)
    return df_analise


# -------------------------------------------------------
# 4. Caracterização demográfica
# -------------------------------------------------------

def plotar_caracterizacao_demografica(df):
    """
    Gera 4 gráficos demográficos sobre o dataset usando seaborn.
    Os dados estão normalizados entre 0 e 1.
    O foco é descrever quem são as comunidades, não explicar o crime.
    """
    df_plot = obter_base_para_analise(df)

    raciais = [c for c in ["racepctblack", "racePctWhite",
                           "racePctAsian", "racePctHisp"] if c in df_plot.columns]
    etarias = [c for c in ["agePct12t21", "agePct16t24",
                           "agePct65up"] if c in df_plot.columns]
    renda_cols = [c for c in ["medIncome", "PctPopUnderPov",
                              "PctUnemployed"] if c in df_plot.columns]
    familia = [c for c in ["PctKids2Par", "MalePctDivorce",
                           "FemalePctDiv"] if c in df_plot.columns]

    # 1. Composição racial média
    if raciais:
        medias = df_plot[raciais].mean().sort_values()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=medias.values, y=medias.index,
                    orient="h", color="steelblue", ax=ax)
        ax.set_title("Composição racial média das comunidades")
        ax.set_xlabel("Proporção média (0-1)")
        ax.set_ylabel("Grupo racial")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "demografico_racial_regressao.png", dpi=150)
        plt.close()
        print("salvo: demografico_racial_regressao.png")

    # 2. Distribuição de renda e pobreza
    if renda_cols:
        n = len(renda_cols)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, col in zip(axes, renda_cols):
            sns.histplot(data=df_plot, x=col, kde=True,
                         color="steelblue", ax=ax)
            ax.set_title(col)
            ax.set_xlabel("Valor (0-1)")
        fig.suptitle("Distribuição de renda e pobreza")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "demografico_renda_regressao.png", dpi=150)
        plt.close()
        print("salvo: demografico_renda_regressao.png")

    # 3. Pirâmide etária simplificada
    if etarias:
        medias = df_plot[etarias].mean().sort_values()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=medias.values, y=medias.index,
                    orient="h", color="coral", ax=ax)
        ax.set_title("Distribuição etária média das comunidades")
        ax.set_xlabel("Proporção média (0-1)")
        ax.set_ylabel("Faixa etária")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "demografico_etario_regressao.png", dpi=150)
        plt.close()
        print("salvo: demografico_etario_regressao.png")

    # 4. Estrutura familiar
    if familia:
        medias = df_plot[familia].mean()
        df_fam = pd.DataFrame(
            {"indicador": medias.index, "media": medias.values})
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=df_fam, x="indicador", y="media", color="teal", ax=ax)
        ax.set_title("Estrutura familiar média das comunidades")
        ax.set_xlabel("Indicador")
        ax.set_ylabel("Proporção média (0-1)")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "demografico_familia_regressao.png", dpi=150)
        plt.close()
        print("salvo: demografico_familia_regressao.png")


# -------------------------------------------------------
# 5. Análise da variável alvo e seleção de features
# -------------------------------------------------------

def plotar_distribuicao_target(df):
    """
    Plota a distribuição da variável alvo ViolentCrimesPerPop.
    Como é uma variável numérica, usa-se histograma com curva de densidade.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x=TARGET, kde=True, color="steelblue", ax=ax)
    ax.set_title("Distribuição da variável alvo - ViolentCrimesPerPop")
    ax.set_xlabel("Crimes violentos por população")
    ax.set_ylabel("Frequência")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "distribuicao_target_regressao.png", dpi=150)
    plt.close()

    print("salvo: distribuicao_target_regressao.png")


def analisar_correlacoes(df):
    """
    Calcula a correlação de Pearson entre cada atributo e a variável alvo.
    Retorna uma tabela ordenada pela correlação absoluta, destacando os atributos
    com maior relação com ViolentCrimesPerPop.
    """
    df_analise = obter_base_para_analise(df)

    correlacoes = df_analise.corr(numeric_only=True)[TARGET].drop(TARGET)
    tabela_corr = pd.DataFrame({
        "correlacao": correlacoes,
        "correlacao_abs": correlacoes.abs()
    }).sort_values("correlacao_abs", ascending=False)

    print("\n--- Top 15 atributos mais correlacionados com a variável alvo ---")
    print(tabela_corr.head(15).round(4))

    return tabela_corr


def plotar_top_correlacoes(tabela_corr, top_n=15):
    """
    Plota os atributos com maior correlação absoluta com a variável alvo.
    """
    df_plot = tabela_corr.head(top_n).copy()
    df_plot = df_plot.reset_index()
    df_plot.columns = ["atributo", "correlacao", "correlacao_abs"]

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.barplot(
        data=df_plot,
        y="atributo",
        x="correlacao_abs",
        color="steelblue",
        ax=ax
    )

    ax.set_title(
        f"Top {top_n} atributos com maior correlação com ViolentCrimesPerPop")
    ax.set_xlabel("Correlação absoluta")
    ax.set_ylabel("Atributo")

    for i, valor in enumerate(df_plot["correlacao_abs"]):
        ax.text(valor + 0.01, i, f"{valor:.4f}", va="center", fontsize=9)

    ax.set_xlim(0, df_plot["correlacao_abs"].max() + 0.08)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "top_15_correlacoes_regressao.png", dpi=150)
    plt.close()

    print("salvo: top_15_correlacoes_regressao.png")


def selecionar_features(df, tabela_corr, limiar=0.5042):
    """
    Seleciona todos os atributos com correlação absoluta >= limiar com a variável alvo.
    Informa o número de features selecionadas.
    """
    features_filtradas = tabela_corr[tabela_corr["correlacao_abs"] >= limiar]
    selected_features = features_filtradas.index.tolist()

    X = df[selected_features]
    y = df[TARGET]

    print(
        f"\nFeatures selecionadas (|correlação| >= {limiar}): {len(selected_features)}")
    print(selected_features)

    return X, y, selected_features


def relatorio_missing_features(df, selected_features):
    """
    Gera relatório de valores faltantes apenas nas features selecionadas + target.
    """
    cols = [TARGET] + [f for f in selected_features if f != TARGET]
    sub = df[cols]

    missing = sub.isna().sum()
    pct = (missing / len(sub) * 100).round(2)

    relatorio = pd.DataFrame({
        "missing": missing,
        "percentual": pct,
        "total": len(sub),
    }).sort_values("missing", ascending=False)

    print("\n--- Valores faltantes nas features selecionadas ---")
    print(f"Total de linhas: {len(sub)}")
    print(relatorio.to_string())

    relatorio.to_csv(CSV_DIR / "missing_features_selecionadas.csv")
    print("salvo: missing_features_selecionadas.csv")

    return relatorio


def plotar_heatmap_correlacao(df, features):
    """
    Plota um heatmap com a variável alvo e as features selecionadas.
    O objetivo é visualizar a relação entre as variáveis escolhidas.
    """
    df_analise = obter_base_para_analise(df)
    cols = features + [TARGET]

    corr = df_analise[cols].corr()

    n = len(cols)
    fig, ax = plt.subplots(figsize=(n * 0.55, n * 0.42))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        annot_kws={"size": 9},
        ax=ax
    )
    ax.set_title("Heatmap de correlação — features selecionadas", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                       ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "heatmap_correlacao_regressao.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    print("salvo: heatmap_correlacao_regressao.png")


def plotar_histogramas_features(df, selected_features):
    """
    Plota histogramas de todas as features selecionadas.
    O primeiro histograma é sempre o da variável alvo ViolentCrimesPerPop.
    """
    df_analise = obter_base_para_analise(df)

    cols = [TARGET] + [f for f in selected_features if f != TARGET]
    n = len(cols)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.histplot(data=df_analise, x=col, kde=True,
                     color="steelblue", ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_xlabel("Valor")
        axes[i].set_ylabel("Frequência")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Histogramas das features selecionadas", y=1.01)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "histogramas_features_regressao.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    print("salvo: histogramas_features_regressao.png")


# -------------------------------------------------------
# Transformações de features
# -------------------------------------------------------

def plotar_boxplots_features(df, selected_features):
    """
    Plota boxplots de todas as features selecionadas + target antes das transformações.
    """
    df_analise = obter_base_para_analise(df)
    cols = [TARGET] + [f for f in selected_features if f != TARGET]

    n = len(cols)
    ncols = 4
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.boxplot(y=df_analise[col], color="steelblue", ax=axes[i])
        axes[i].set_title(col, fontsize=9)
        axes[i].set_ylabel("")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        "Boxplots das features selecionadas (pré-transformação)", y=1.01)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "boxplots_features_regressao.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    print("salvo: boxplots_features_regressao.png")


def aplicar_transformacoes(df, transformacoes):
    """
    Aplica transformações log (log1p) ou power (x²) nas colunas indicadas.
    Parte de base já imputada para evitar NaN nas transformações.
    """
    df_t = obter_base_para_analise(df).copy()

    for col, tipo in transformacoes.items():
        if col not in df_t.columns:
            continue
        if tipo == "log":
            df_t[col] = np.log1p(df_t[col])
        elif tipo == "power":
            df_t[col] = df_t[col] ** 2

    transformadas = [c for c in transformacoes if c in df_t.columns]
    print(f"\nTransformações aplicadas em {len(transformadas)} colunas.")
    return df_t


def plotar_histogramas_transformados(df_transf, selected_features, transformacoes):
    """
    Plota histogramas das features selecionadas após transformação.
    Subtítulo de cada painel indica o tipo de transformação aplicada.
    """
    cols = [TARGET] + [f for f in selected_features if f != TARGET]
    n = len(cols)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        tipo = transformacoes.get(col, "original")
        sns.histplot(data=df_transf, x=col, kde=True,
                     color="coral", ax=axes[i])
        axes[i].set_title(f"{col}\n({tipo})")
        axes[i].set_xlabel("Valor transformado")
        axes[i].set_ylabel("Frequência")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Histogramas após transformações", y=1.01)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "histogramas_transformados_regressao.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    print("salvo: histogramas_transformados_regressao.png")


# -------------------------------------------------------
# 6. Preparação para modelagem
# -------------------------------------------------------

def dividir_dados(X, y):
    """
    Divide os dados em treino e teste.

    Utiliza 80% dos dados para treino e 20% para teste.
    O random_state garante que a divisão seja reproduzível.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\n--- Divisão dos dados ---")
    print(f"Treino: {X_train.shape[0]} amostras")
    print(f"Teste: {X_test.shape[0]} amostras")

    return X_train, X_test, y_train, y_test


def construir_pipeline(modelo):
    """
    Constrói um Pipeline para regressão.

    Etapas:
    - SimpleImputer: preenche valores ausentes usando a mediana do treino.
    - StandardScaler: padroniza os atributos.
    - Modelo: LinearRegression, Ridge ou Lasso.

    O uso de Pipeline evita data leakage porque o tratamento é ajustado apenas
    nos dados de treino.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", modelo)
    ])

    return pipe


# -------------------------------------------------------
# 7. Treinamento e avaliação dos modelos
# -------------------------------------------------------

def calcular_metricas(y_true, y_pred):
    """
    Calcula as principais métricas de regressão exigidas no trabalho:
    MAE, MSE, RMSE e R².
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


def treinar_modelos(X_train, X_test, y_train, y_test):
    """
    Treina três modelos de regressão:
    - Regressão Linear padrão;
    - Ridge Regression;
    - Lasso Regression.

    Retorna os modelos treinados, as predições e a tabela de métricas.
    """
    modelos = {
        "Regressão Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=10000)
    }

    modelos_treinados = {}
    predicoes = {}
    resultados = []

    for nome, modelo in modelos.items():
        pipe = construir_pipeline(modelo)
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        modelos_treinados[nome] = pipe
        predicoes[nome] = y_pred

        metricas = calcular_metricas(y_test, y_pred)
        resultados.append({
            "modelo": nome,
            "MAE": metricas["MAE"],
            "MSE": metricas["MSE"],
            "RMSE": metricas["RMSE"],
            "R2": metricas["R2"]
        })

    df_resultados = pd.DataFrame(resultados).set_index("modelo")

    print("\n--- Resultados no conjunto de teste ---")
    print(df_resultados.round(4))

    df_resultados.to_csv(CSV_DIR / "resultados_teste_regressao.csv")

    return modelos_treinados, predicoes, df_resultados


def _plotar_comparacao_agrupada(df_resultados, col_map, nome_arquivo, titulo):
    """
    Gera gráfico de barras agrupadas por modelo com MAE, MSE e RMSE.
    R² é exibido no rótulo do eixo X: "Modelo (R² = xx)".
    Valor numérico é anotado no topo de cada barra.
    """
    df_plot = df_resultados.reset_index().rename(columns={"modelo": "Modelo"})
    df_plot = df_plot.rename(columns=col_map)

    # R² vai para o label do modelo, não para as barras
    r2_por_modelo = df_plot.set_index("Modelo")["R²"]
    df_plot["Modelo_label"] = df_plot["Modelo"].apply(
        lambda m: f"{m}\n(R² = {r2_por_modelo[m]:.4f})"
    )

    metricas = [v for v in col_map.values() if v != "R²"]

    df_melt = df_plot.melt(
        id_vars="Modelo_label",
        value_vars=metricas,
        var_name="Métrica",
        value_name="Valor"
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=df_melt, x="Modelo_label",
                y="Valor", hue="Métrica", ax=ax)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=3, fontsize=8)

    ax.set_title(f"Comparação dos modelos — {titulo}")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Valor")
    ax.legend(title="Métrica", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / nome_arquivo, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"salvo: {nome_arquivo}")


def plotar_comparacao_modelos(df_resultados):
    """
    Gráfico de barras agrupadas comparando MAE, MSE, RMSE e R²
    dos três modelos no conjunto de teste.
    """
    _plotar_comparacao_agrupada(
        df_resultados,
        {"MSE": "MSE", "MAE": "MAE", "RMSE": "RMSE", "R2": "R²"},
        "comparacao_treino_regressao.png",
        "Treino Normal"
    )


def plotar_comparacao_cv(df_cv):
    """
    Gráfico de barras agrupadas comparando MAE, MSE, RMSE e R²
    dos três modelos na validação cruzada. Mesmo estilo do treino normal.
    """
    _plotar_comparacao_agrupada(
        df_cv,
        {"MSE_CV": "MSE", "MAE_CV": "MAE", "RMSE_CV": "RMSE", "R2_CV": "R²"},
        "comparacao_cv_regressao.png",
        "Validação Cruzada"
    )


# -------------------------------------------------------
# 8. Regressão linear simples com um atributo
# -------------------------------------------------------

def escolher_atributo_regressao_simples(tabela_corr):
    """
    Escolhe automaticamente o atributo com maior correlação absoluta com a variável alvo.
    Esse atributo será usado para visualizar a reta de regressão simples.
    """
    atributo = tabela_corr.index[0]
    correlacao = tabela_corr.iloc[0]["correlacao"]

    print("\n--- Atributo escolhido para regressão simples ---")
    print(f"Atributo: {atributo}")
    print(f"Correlação com {TARGET}: {correlacao:.4f}")

    return atributo


def plotar_reta_regressao(df_transf, atributo, df_grid, resultados_grid):
    """
    Retreina o melhor modelo do Grid Search usando apenas o atributo mais
    correlacionado e plota scatter + reta de regressão.
    Título inclui todos os hiperparâmetros do modelo escolhido.
    """
    nome_melhor = df_grid["melhor_R2_CV"].idxmax()

    gs = resultados_grid[nome_melhor]
    best_params = {k.replace("modelo__", ""): v for k, v in gs.best_params_.items()}

    pipe = clone(gs.best_estimator_)

    dados = df_transf[[atributo, TARGET]].copy().dropna()
    X = dados[[atributo]]
    y = dados[TARGET]

    pipe.fit(X, y)

    x_linha = np.linspace(X[atributo].min(), X[atributo].max(), 200)
    y_linha = pipe.predict(pd.DataFrame({atributo: x_linha}))

    params_str = " | ".join(f"{k}={v}" for k, v in best_params.items())

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(data=dados, x=atributo, y=TARGET, alpha=0.5, ax=ax)
    ax.plot(x_linha, y_linha, linewidth=2, color="crimson")
    ax.set_title(f"{nome_melhor}: {atributo} × {TARGET}\n{params_str}", fontsize=10)
    ax.set_xlabel(atributo)
    ax.set_ylabel(TARGET)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "reta_regressao.png", dpi=150)
    plt.close()

    print("salvo: reta_regressao.png")


def plotar_retas_todas_features_melhor_grid(df_transf, selected_features, df_grid, tabela_corr, resultados_grid):
    """
    Para o melhor modelo do Grid Search, gera um subplot por feature:
    scatter real + linha de efeito parcial (varia a feature, fixa as demais na média).
    Título de cada subplot inclui correlação com a variável alvo.
    """
    nome_melhor = df_grid["melhor_R2_CV"].idxmax()
    info = df_grid.loc[nome_melhor]

    gs = resultados_grid[nome_melhor]
    best_params = {k.replace("modelo__", ""): v for k, v in gs.best_params_.items()}

    pipe = clone(gs.best_estimator_)

    dados = df_transf[selected_features + [TARGET]].dropna()
    X_all = dados[selected_features]
    y_all = dados[TARGET]

    pipe.fit(X_all, y_all)

    medias = X_all.mean()
    n = len(selected_features)
    ncols = 4
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, feat in enumerate(selected_features):
        corr_val = tabela_corr.loc[feat, "correlacao"] if feat in tabela_corr.index else float(
            "nan")

        x_range = np.linspace(X_all[feat].min(), X_all[feat].max(), 200)
        X_partial = pd.DataFrame(
            np.tile(medias.values, (200, 1)),
            columns=selected_features
        )
        X_partial[feat] = x_range
        y_linha = pipe.predict(X_partial)

        ax = axes[i]
        sns.scatterplot(x=X_all[feat], y=y_all, alpha=0.3,
                        s=10, color="steelblue", ax=ax)
        ax.plot(x_range, y_linha, color="crimson", linewidth=1.8)
        ax.set_title(f"{feat}\nr={corr_val:.3f}", fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel(TARGET if i % ncols == 0 else "")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    params_str = " | ".join(f"{k}={v}" for k, v in best_params.items())
    fig.suptitle(
        f"Efeito parcial por feature — {nome_melhor}  |  R²={info['R2_teste']:.4f}\n{params_str}",
        fontsize=10, y=1.01
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "retas_features_melhor_grid_regressao.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    print("salvo: retas_features_melhor_grid_regressao.png")


def plotar_reta_melhor_grid(df_transf, atributo, df_grid, resultados_grid):
    """
    Retreina o melhor modelo do Grid Search usando apenas o atributo mais correlacionado
    e plota scatter + reta, com caixa de anotação contendo configs e métricas.
    """
    nome_melhor = df_grid["melhor_R2_CV"].idxmax()
    info = df_grid.loc[nome_melhor]

    gs = resultados_grid[nome_melhor]
    best_params = {k.replace("modelo__", ""): v for k, v in gs.best_params_.items()}
    alpha = best_params.get("alpha")

    pipe = clone(gs.best_estimator_)

    dados = df_transf[[atributo, TARGET]].copy().dropna()
    X_s = dados[[atributo]]
    y_s = dados[TARGET]

    pipe.fit(X_s, y_s)
    y_pred = pipe.predict(X_s)

    metricas = calcular_metricas(y_s, y_pred)

    x_linha = np.linspace(X_s[atributo].min(), X_s[atributo].max(), 200)
    y_linha = pipe.predict(pd.DataFrame({atributo: x_linha}))

    alpha_str = f"α = {alpha}" if alpha is not None else "sem regularização"
    legenda = (
        f"Modelo: {nome_melhor}\n"
        f"{alpha_str}\n"
        f"R² (CV grid): {info['melhor_R2_CV']:.4f}\n"
        f"R² teste:     {metricas['R2']:.4f}\n"
        f"MAE:          {metricas['MAE']:.4f}\n"
        f"MSE:          {metricas['MSE']:.4f}\n"
        f"RMSE:         {metricas['RMSE']:.4f}"
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=dados, x=atributo, y=TARGET, alpha=0.45, ax=ax)
    ax.plot(x_linha, y_linha, linewidth=2,
            color="crimson", label="reta ajustada")
    ax.set_title(
        f"Melhor modelo (Grid Search): {nome_melhor}\n{atributo} × {TARGET}")
    ax.set_xlabel(atributo)
    ax.set_ylabel(TARGET)
    ax.text(
        0.97, 0.05, legenda,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="gray", alpha=0.85),
        fontfamily="monospace",
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "reta_melhor_grid_regressao.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    print("salvo: reta_melhor_grid_regressao.png")


# -------------------------------------------------------
# 9. Validação cruzada
# -------------------------------------------------------

def validacao_cruzada(X_train, y_train):
    """
    Aplica validação cruzada 5-fold para avaliar os modelos de forma mais robusta.

    São calculadas as métricas:
    - MAE;
    - MSE;
    - RMSE;
    - R².

    As métricas negativas retornadas pelo scikit-learn são convertidas para valores
    positivos quando representam erro.
    """
    modelos = {
        "Regressão Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=10000)
    }

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "MAE": "neg_mean_absolute_error",
        "MSE": "neg_mean_squared_error",
        "R2": "r2"
    }

    resultados_cv = []

    for nome, modelo in modelos.items():
        pipe = construir_pipeline(modelo)

        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=kfold,
            scoring=scoring,
            n_jobs=1
        )

        mae = -scores["test_MAE"].mean()
        mse = -scores["test_MSE"].mean()
        rmse = np.sqrt(mse)
        r2 = scores["test_R2"].mean()

        resultados_cv.append({
            "modelo": nome,
            "MAE_CV": mae,
            "MSE_CV": mse,
            "RMSE_CV": rmse,
            "R2_CV": r2
        })

    df_cv = pd.DataFrame(resultados_cv).set_index("modelo")

    print("\n--- Validação Cruzada 5-fold ---")
    print(df_cv.round(4))

    df_cv.to_csv(CSV_DIR / "resultados_validacao_cruzada.csv")

    return df_cv, kfold


# -------------------------------------------------------
# 10. Grid Search
# -------------------------------------------------------

def grid_search(X_train, X_test, y_train, y_test, kfold):
    """
    Aplica Grid Search em todos os três modelos, varrendo todos os hiperparâmetros
    relevantes de cada um via validação cruzada.

    LinearRegression: fit_intercept
    Ridge:            alpha, fit_intercept, solver
    Lasso:            alpha, fit_intercept, selection

    O melhor modelo é escolhido com base no maior R² médio na validação cruzada.
    """
    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

    configs = {
        "Regressão Linear": {
            "modelo": LinearRegression(),
            "param_grid": {
                "modelo__fit_intercept": [True, False],
            },
        },
        "Ridge": {
            "modelo": Ridge(),
            "param_grid": {
                "modelo__alpha": alphas,
                "modelo__fit_intercept": [True, False],
                "modelo__solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
            },
        },
        "Lasso": {
            "modelo": Lasso(max_iter=20000),
            "param_grid": {
                "modelo__alpha": alphas,
                "modelo__fit_intercept": [True, False],
                "modelo__selection": ["cyclic", "random"],
            },
        },
    }

    resultados_grid = {}
    linhas = []

    for nome, config in configs.items():
        pipe = construir_pipeline(config["modelo"])

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=config["param_grid"],
            cv=kfold,
            scoring="r2",
            n_jobs=-1,
        )

        gs.fit(X_train, y_train)

        y_pred = gs.predict(X_test)
        metricas = calcular_metricas(y_test, y_pred)
        resultados_grid[nome] = gs

        best = {k.replace("modelo__", ""): v for k, v in gs.best_params_.items()}

        linhas.append({
            "modelo": nome,
            "melhor_R2_CV": gs.best_score_,
            "MAE_teste": metricas["MAE"],
            "MSE_teste": metricas["MSE"],
            "RMSE_teste": metricas["RMSE"],
            "R2_teste": metricas["R2"],
            **best,
        })

    df_grid = pd.DataFrame(linhas).set_index("modelo")

    print("\n--- Grid Search completo ---")
    print(df_grid.round(4))

    df_grid.to_csv(CSV_DIR / "resultados_grid_search.csv")

    return resultados_grid, df_grid


# -------------------------------------------------------
# 11. Comparação final
# -------------------------------------------------------

def comparar_resultados(df_teste, df_cv, df_grid):
    """
    Junta os resultados de treino/teste, validação cruzada e Grid Search.
    Essa tabela facilita a comparação final entre os modelos.
    """
    comparacao = df_teste.copy()
    comparacao = comparacao.join(df_cv, how="left")

    print("\n--- Comparação geral: teste x validação cruzada ---")
    print(comparacao.round(4))

    print("\n--- Resultados otimizados por Grid Search ---")
    print(df_grid.round(4))

    comparacao.to_csv(CSV_DIR / "comparacao_geral_regressao.csv")

    return comparacao


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    """Executa o pipeline completo de regressão da AV2."""

    PLOTS_DIR.mkdir(exist_ok=True)
    CSV_DIR.mkdir(exist_ok=True)

    # 1. Carregamento e inspeção inicial
    df = carregar_dados()
    explorar_dados(df)

    # 2. Pré-processamento: remove colunas com >= 50% de valores ausentes
    tabela_missing = verificar_missing(df)
    df_limpo, _, cols_categoricas = preparar_base(df)

    df_limpo = codificar_categoricas(df_limpo, cols_categoricas)

    plotar_distribuicao_target(df_limpo)

    # 3. Caracterização demográfica
    plotar_caracterizacao_demografica(df_limpo)

    # 4. Seleção de features com correlação >= 0.7
    tabela_corr = analisar_correlacoes(df_limpo)

    _, _, selected_features = selecionar_features(
        df_limpo, tabela_corr, limiar=0.5042)

    relatorio_missing_features(df_limpo, selected_features)

    plotar_heatmap_correlacao(df_limpo, selected_features)
    plotar_histogramas_features(df_limpo, selected_features)

    # 4b. Boxplots pré-transformação + transformações + histogramas atualizados
    plotar_boxplots_features(df_limpo, selected_features)
    df_transf = aplicar_transformacoes(df_limpo, TRANSFORMACOES)
    plotar_histogramas_transformados(
        df_transf, selected_features, TRANSFORMACOES)

    # 5. Modelagem com dados transformados
    X_transf = df_transf[selected_features]
    y_transf = df_transf[TARGET]
    X_train, X_test, y_train, y_test = dividir_dados(X_transf, y_transf)

    _, _, df_teste = treinar_modelos(X_train, X_test, y_train, y_test)

    # 5a. Gráfico comparativo — treino normal
    plotar_comparacao_modelos(df_teste)

    atributo_simples = escolher_atributo_regressao_simples(tabela_corr)

    df_cv, kfold = validacao_cruzada(X_train, y_train)

    # 5b. Gráfico comparativo — validação cruzada (mesmo estilo)
    plotar_comparacao_cv(df_cv)

    resultados_grid, df_grid = grid_search(X_train, X_test, y_train, y_test, kfold)

    plotar_reta_regressao(df_transf, atributo_simples, df_grid, resultados_grid)
    plotar_retas_todas_features_melhor_grid(
        df_transf, selected_features, df_grid, tabela_corr, resultados_grid)

    comparar_resultados(df_teste, df_cv, df_grid)

    print("\nConcluído.")


if __name__ == "__main__":
    main()
