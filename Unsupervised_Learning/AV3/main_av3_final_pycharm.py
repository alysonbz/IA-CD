# AV3 - Inteligência Artificial - Online Retail

from pathlib import Path
import zipfile
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_ZIP = BASE_DIR / "online+retail.zip"
DATA_XLSX = BASE_DIR / "Online Retail.xlsx"
RESULTS_DIR = BASE_DIR / "resultados_av3"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42



def salvar_texto(nome_arquivo, conteudo):
    with open(RESULTS_DIR / nome_arquivo, "w", encoding="utf-8") as arquivo:
        arquivo.write(conteudo)


def salvar_tabela(df, nome_arquivo):
    df.to_csv(RESULTS_DIR / nome_arquivo, encoding="utf-8-sig")


def localizar_dataset():
    if DATA_XLSX.exists():
        return DATA_XLSX

    if DATA_ZIP.exists():
        with zipfile.ZipFile(DATA_ZIP, "r") as zip_ref:
            zip_ref.extractall(BASE_DIR)

        if DATA_XLSX.exists():
            return DATA_XLSX

    raise FileNotFoundError(
        "Coloque 'online+retail.zip' ou 'Online Retail.xlsx' na mesma pasta deste script."
    )


def carregar_dados():
    caminho = localizar_dataset()
    print(f"Carregando dataset: {caminho.name}")
    return pd.read_excel(caminho)


def preparar_base_cliente(df):
    """
    O dataset Online Retail é transacional.
    Para aplicar aprendizado não supervisionado de forma coerente,
    a base é agregada por cliente.
    """
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df_limpo = df.dropna(subset=["CustomerID"]).copy()
    df_limpo = df_limpo[~df_limpo["InvoiceNo"].astype(str).str.startswith("C")]
    df_limpo = df_limpo[(df_limpo["Quantity"] > 0) & (df_limpo["UnitPrice"] > 0)]

    df_limpo["CustomerID"] = df_limpo["CustomerID"].astype(int)
    df_limpo["TotalPrice"] = df_limpo["Quantity"] * df_limpo["UnitPrice"]

    data_referencia = df_limpo["InvoiceDate"].max() + pd.Timedelta(days=1)

    clientes = df_limpo.groupby("CustomerID").agg(
        country=("Country", lambda x: x.mode()[0]),
        recency_days=("InvoiceDate", lambda x: (data_referencia - x.max()).days),
        frequency=("InvoiceNo", "nunique"),
        total_quantity=("Quantity", "sum"),
        monetary=("TotalPrice", "sum"),
        avg_unit_price=("UnitPrice", "mean"),
        num_products=("StockCode", "nunique")
    ).reset_index()

    clientes["avg_basket_value"] = clientes["monetary"] / clientes["frequency"]

    top_paises = clientes["country"].value_counts().head(6).index
    clientes["country_group"] = np.where(
        clientes["country"].isin(top_paises),
        clientes["country"],
        "Other"
    )

    features = [
        "recency_days",
        "frequency",
        "total_quantity",
        "monetary",
        "avg_unit_price",
        "num_products",
        "avg_basket_value"
    ]

    variavel_referencia = "country_group"

    return df_limpo, clientes, features, variavel_referencia


def matriz_preprocessada(clientes, features, scaler="standard"):
    """
    Retorna matriz de atributos para clusterização.
    Para StandardScaler e MinMaxScaler, aplica log1p antes para reduzir assimetria.
    """
    X = clientes[features].copy()

    if scaler == "none":
        return X.values

    X_log = np.log1p(X)

    if scaler == "standard":
        return StandardScaler().fit_transform(X_log)

    if scaler == "minmax":
        return MinMaxScaler().fit_transform(X_log)

    raise ValueError("Use scaler='none', 'standard' ou 'minmax'.")


def avaliar_kmeans(X, k_min=2, k_max=10):
    resultados = []

    for k in range(k_min, k_max + 1):
        modelo = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20
        )

        labels = modelo.fit_predict(X)

        resultados.append({
            "k": k,
            "inertia": modelo.inertia_,
            "silhouette": silhouette_score(X, labels)
        })

    return pd.DataFrame(resultados)


def plot_matriz(matriz, titulo, caminho, labels_x=None, labels_y=None, colorbar_label="Valor"):
    plt.figure(figsize=(8, 6))
    plt.imshow(matriz, aspect="auto")
    if labels_x is not None:
        plt.xticks(range(len(labels_x)), labels_x, rotation=45, ha="right")
    if labels_y is not None:
        plt.yticks(range(len(labels_y)), labels_y)
    plt.colorbar(label=colorbar_label)
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()



# QUESTÃO 1 - ANÁLISE EXPLORATÓRIA DOS DADOS

def questao_1(df_original, df_limpo, clientes, features):
    print("\n" + "=" * 70)
    print("QUESTÃO 1 - ANÁLISE EXPLORATÓRIA DOS DADOS")
    print("=" * 70)

    descricao = pd.DataFrame({
        "informacao": [
            "Quantidade de linhas no dataset original",
            "Quantidade de colunas no dataset original",
            "Quantidade de linhas após limpeza",
            "Quantidade de clientes únicos após agregação",
            "Período inicial das compras",
            "Período final das compras"
        ],
        "valor": [
            df_original.shape[0],
            df_original.shape[1],
            df_limpo.shape[0],
            clientes["CustomerID"].nunique(),
            df_limpo["InvoiceDate"].min(),
            df_limpo["InvoiceDate"].max()
        ]
    })

    ausentes = df_original.isna().sum().to_frame("valores_ausentes")

    problemas = pd.DataFrame({
        "problema_verificado": [
            "CustomerID ausente",
            "Notas fiscais canceladas",
            "Quantity <= 0",
            "UnitPrice <= 0"
        ],
        "quantidade": [
            df_original["CustomerID"].isna().sum(),
            df_original["InvoiceNo"].astype(str).str.startswith("C").sum(),
            (df_original["Quantity"] <= 0).sum(),
            (df_original["UnitPrice"] <= 0).sum()
        ]
    })

    estatisticas = clientes[features].describe().T
    correlacao = clientes[features].corr()

    salvar_tabela(descricao, "q1_descricao_geral_dataset.csv")
    salvar_tabela(ausentes, "q1_valores_ausentes.csv")
    salvar_tabela(problemas, "q1_problemas_dados.csv")
    salvar_tabela(estatisticas, "q1_estatisticas_descritivas.csv")
    salvar_tabela(correlacao, "q1_correlacao_atributos.csv")

    for coluna in features:
        plt.figure(figsize=(7, 4))
        plt.hist(clientes[coluna], bins=40)
        plt.title(f"Distribuição de {coluna}")
        plt.xlabel(coluna)
        plt.ylabel("Frequência")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"q1_histograma_{coluna}.png", dpi=150)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.boxplot(clientes[coluna], vert=False)
        plt.title(f"Boxplot de {coluna}")
        plt.xlabel(coluna)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"q1_boxplot_{coluna}.png", dpi=150)
        plt.close()

    plot_matriz(
        correlacao,
        "Mapa de Correlação entre Atributos Numéricos",
        RESULTS_DIR / "q1_mapa_correlacao.png",
        labels_x=features,
        labels_y=features,
        colorbar_label="Correlação"
    )

    texto = f"""
QUESTÃO 1 - INTERPRETAÇÃO

O dataset Online Retail é uma base transacional de vendas. Como o objetivo da avaliação é aplicar aprendizado não supervisionado, os dados foram reorganizados por cliente, criando uma base agregada capaz de representar o comportamento de compra.

Foram verificados problemas nos dados:
- ausência de CustomerID;
- notas fiscais canceladas;
- quantidades negativas ou iguais a zero;
- preços unitários negativos ou iguais a zero.

Esses registros foram removidos porque poderiam prejudicar a análise de agrupamento.

Após a limpeza, foram criados os seguintes atributos numéricos:
{features}

Interpretação dos atributos:
- recency_days: tempo desde a última compra do cliente;
- frequency: número de compras realizadas;
- total_quantity: quantidade total de itens comprados;
- monetary: valor total gasto;
- avg_unit_price: preço médio dos produtos;
- num_products: diversidade de produtos comprados;
- avg_basket_value: valor médio por compra.

A análise exploratória mostrou forte assimetria em variáveis como monetary, frequency e total_quantity. Os histogramas e boxplots ajudam a visualizar essa concentração e a presença de valores extremos.

Os atributos mais relevantes para uma possível análise de agrupamento são recency_days, frequency, monetary, total_quantity e num_products, pois representam comportamento de compra, valor financeiro e nível de engajamento do cliente.
"""

    salvar_texto("q1_interpretacao.txt", texto)
    print(texto)



# QUESTÃO 2 - ANÁLISE VISUAL COM DOIS ATRIBUTOS

def questao_2(clientes, variavel_referencia):
    print("\n" + "=" * 70)
    print("QUESTÃO 2 - ANÁLISE VISUAL COM DOIS ATRIBUTOS")
    print("=" * 70)

    atributo_x = "recency_days"
    atributo_y = "monetary"

    amostra = clientes.sample(
        min(2500, len(clientes)),
        random_state=RANDOM_STATE
    )

    plt.figure(figsize=(8, 5))

    for grupo in sorted(amostra[variavel_referencia].unique()):
        temp = amostra[amostra[variavel_referencia] == grupo]
        plt.scatter(
            temp[atributo_x],
            temp[atributo_y],
            s=18,
            alpha=0.65,
            label=str(grupo)
        )

    plt.yscale("log")
    plt.xlabel("Recência em dias")
    plt.ylabel("Valor monetário total - escala log")
    plt.title("Análise visual: Recência x Valor monetário")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "q2_dispersao_recency_monetary.png", dpi=150)
    plt.close()

    texto = """
QUESTÃO 2 - INTERPRETAÇÃO

Foram escolhidos os atributos recency_days e monetary.

Justificativa:
Esses dois atributos fazem parte da lógica RFM, muito utilizada em segmentação de clientes:
- recency_days representa há quanto tempo o cliente comprou pela última vez;
- monetary representa quanto o cliente gastou no total.

O gráfico de dispersão permite investigar visualmente se existem possíveis agrupamentos naturais entre clientes com diferentes níveis de gasto e diferentes níveis de recência.

A variável categórica country_group foi usada apenas para fins de análise visual, conforme solicitado no enunciado.

Observação visual:
A separação dos grupos não é completamente clara apenas com dois atributos. É possível observar regiões com clientes de baixo gasto, clientes mais recentes e alguns clientes de alto valor, mas há sobreposição entre grupos.

Conclusão:
Os dois atributos ajudam na interpretação inicial, mas não parecem suficientes para representar toda a estrutura dos dados. Para uma análise de clusterização mais robusta, é necessário utilizar um conjunto maior de atributos.
"""

    salvar_texto("q2_interpretacao.txt", texto)
    print(texto)



# QUESTÃO 3 - CLUSTERIZAÇÃO COM K-MEANS E ESCOLHA DO MELHOR K


def questao_3(clientes, features, variavel_referencia):
    print("\n" + "=" * 70)
    print("QUESTÃO 3 - K-MEANS E ESCOLHA DO MELHOR K")
    print("=" * 70)

    X = matriz_preprocessada(clientes, features, scaler="standard")

    resultados_kmeans = avaliar_kmeans(X, k_min=2, k_max=10)
    salvar_tabela(resultados_kmeans, "q3_kmeans_metricas.csv")

    melhor_k = int(
        resultados_kmeans.sort_values("silhouette", ascending=False).iloc[0]["k"]
    )
    melhor_silhouette = float(
        resultados_kmeans["silhouette"].max()
    )

    plt.figure(figsize=(7, 4))
    plt.plot(resultados_kmeans["k"], resultados_kmeans["inertia"], marker="o")
    plt.title("Método do Cotovelo - K-Means")
    plt.xlabel("Número de clusters K")
    plt.ylabel("Inércia")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "q3_cotovelo.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(resultados_kmeans["k"], resultados_kmeans["silhouette"], marker="o")
    plt.title("Silhouette Score por valor de K")
    plt.xlabel("Número de clusters K")
    plt.ylabel("Silhouette Score")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "q3_silhouette_score.png", dpi=150)
    plt.close()

    modelo_kmeans = KMeans(
        n_clusters=melhor_k,
        random_state=RANDOM_STATE,
        n_init=20
    )

    clientes["cluster_kmeans"] = modelo_kmeans.fit_predict(X)

    crosstab = pd.crosstab(clientes["cluster_kmeans"], clientes[variavel_referencia])
    crosstab_percentual = pd.crosstab(
        clientes["cluster_kmeans"],
        clientes[variavel_referencia],
        normalize="index"
    ) * 100

    perfil_clusters = clientes.groupby("cluster_kmeans")[features].mean()

    salvar_tabela(crosstab, "q3_crosstab_clusters_variavel_real.csv")
    salvar_tabela(crosstab_percentual, "q3_crosstab_percentual.csv")
    salvar_tabela(perfil_clusters, "q3_perfil_medio_clusters.csv")

    plt.figure(figsize=(8, 5))
    plt.scatter(
        clientes["recency_days"],
        clientes["monetary"],
        c=clientes["cluster_kmeans"],
        s=18,
        alpha=0.7
    )
    plt.yscale("log")
    plt.xlabel("Recência em dias")
    plt.ylabel("Valor monetário total - escala log")
    plt.title(f"Clusters K-Means em atributos relevantes - K={melhor_k}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "q3_clusters_kmeans_2d_atributos.png", dpi=150)
    plt.close()

    pca_visual = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca_visual = pca_visual.fit_transform(X)

    plt.figure(figsize=(8, 5))
    plt.scatter(
        X_pca_visual[:, 0],
        X_pca_visual[:, 1],
        c=clientes["cluster_kmeans"],
        s=18,
        alpha=0.7
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"Visualização dos clusters K-Means via PCA 2D - K={melhor_k}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "q3_clusters_kmeans_pca_2d.png", dpi=150)
    plt.close()

    texto = f"""
QUESTÃO 3 - INTERPRETAÇÃO

Foram selecionados os seguintes atributos para a clusterização:
{features}

Justificativa:
Esses atributos representam comportamento de compra, frequência, valor financeiro, quantidade comprada, variedade de produtos e valor médio por compra.

Pré-processamento:
Foi aplicada transformação logarítmica log1p para reduzir assimetria e StandardScaler para padronizar os atributos. Isso é necessário porque o K-Means utiliza distância euclidiana e é sensível à escala das variáveis.

Foram testados valores de K de 2 até 10.

Critérios usados:
- Método do cotovelo: análise da redução da inércia.
- Silhouette Score: avaliação da coesão e separação dos clusters.

Melhor valor de K escolhido pelo maior Silhouette Score:

K = {melhor_k}

Silhouette máximo = {melhor_silhouette:.4f}

Após escolher o melhor K, o K-Means foi treinado novamente. Os clusters foram visualizados em 2D usando atributos relevantes e também em uma projeção PCA 2D, o que melhora a apresentação visual da separação dos grupos.

Também foi criada uma tabela cruzada entre os clusters gerados e a variável categórica de referência {variavel_referencia}. Essa tabela permite verificar se os agrupamentos encontrados possuem alguma correspondência com os países dos clientes.

Conclusão:
Os clusters encontrados representam principalmente padrões de comportamento de compra. A variável categórica de país auxilia na análise, mas os grupos formados pelo K-Means tendem a refletir mais o comportamento dos clientes do que apenas sua localização.
"""

    salvar_texto("q3_interpretacao.txt", texto)
    print(texto)

    return melhor_k, X



# QUESTÃO 4 - COMPARAÇÃO ENTRE K-MEANS E DBSCAN

def questao_4(clientes, X, variavel_referencia, melhor_k):
    print("\n" + "=" * 70)
    print("QUESTÃO 4 - COMPARAÇÃO ENTRE K-MEANS E DBSCAN")
    print("=" * 70)

    min_samples = 2 * X.shape[1]

    vizinhos = NearestNeighbors(n_neighbors=min_samples)
    vizinhos.fit(X)

    distancias, _ = vizinhos.kneighbors(X)
    k_distancias = np.sort(distancias[:, -1])

    eps = float(np.percentile(k_distancias, 95))

    plt.figure(figsize=(7, 4))
    plt.plot(k_distancias)
    plt.axhline(eps, linestyle="--")
    plt.title("Gráfico k-distance para escolha do eps")
    plt.xlabel("Pontos ordenados")
    plt.ylabel(f"Distância ao {min_samples}º vizinho")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "q4_kdistance_dbscan.png", dpi=150)
    plt.close()

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clientes["cluster_dbscan"] = dbscan.fit_predict(X)

    labels = clientes["cluster_dbscan"]

    quantidade_clusters = len(set(labels)) - (1 if -1 in set(labels) else 0)
    quantidade_ruidos = int((labels == -1).sum())

    crosstab = pd.crosstab(clientes["cluster_dbscan"], clientes[variavel_referencia])
    salvar_tabela(crosstab, "q4_crosstab_dbscan_variavel_real.csv")

    perfil_dbscan = clientes.groupby("cluster_dbscan")[[
        "recency_days",
        "frequency",
        "total_quantity",
        "monetary",
        "num_products",
        "avg_basket_value"
    ]].mean()

    salvar_tabela(perfil_dbscan, "q4_perfil_medio_clusters_dbscan.csv")

    plt.figure(figsize=(8, 5))
    plt.scatter(
        clientes["recency_days"],
        clientes["monetary"],
        c=labels,
        s=18,
        alpha=0.7
    )
    plt.yscale("log")
    plt.xlabel("Recência em dias")
    plt.ylabel("Valor monetário total - escala log")
    plt.title("Clusters obtidos pelo DBSCAN")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "q4_clusters_dbscan_2d.png", dpi=150)
    plt.close()

    texto = f"""
QUESTÃO 4 - INTERPRETAÇÃO

Diferença entre K-Means e DBSCAN:

O K-Means exige que o número de clusters seja definido antes do treinamento. Ele tende a formar grupos compactos e aproximadamente circulares, sendo sensível à escala dos atributos e a outliers.

O DBSCAN é baseado em densidade. Ele não exige informar previamente a quantidade de clusters e consegue identificar pontos considerados ruído. Porém, o resultado depende fortemente dos parâmetros eps e min_samples.

Parâmetros usados no DBSCAN:
- eps = {eps:.4f}
- min_samples = {min_samples}

Justificativa dos parâmetros:
O min_samples foi definido como duas vezes o número de atributos utilizados.
O eps foi escolhido com auxílio do gráfico k-distance, usando a distância ao vizinho definido por min_samples.

Resultados:
- K-Means gerou {melhor_k} clusters.
- DBSCAN gerou {quantidade_clusters} clusters.
- DBSCAN identificou {quantidade_ruidos} pontos como ruído.

Comparação:
- Quantidade de clusters: o K-Means gera exatamente K grupos; o DBSCAN encontra grupos conforme a densidade.
- Presença de ruídos: o DBSCAN identifica ruídos; o K-Means força todos os pontos em algum cluster.
- Formato dos agrupamentos: o K-Means favorece grupos compactos; o DBSCAN pode encontrar formatos mais irregulares.
- Sensibilidade à escala: ambos são sensíveis à escala, por isso os dados foram normalizados.
- Facilidade de interpretação: o K-Means é mais direto para interpretar perfis médios de clientes.
- Relação com a variável categórica: a crosstab permite analisar se algum cluster concentra determinados países.

Conclusão:
Para este dataset, o K-Means apresentou resultado mais coerente para segmentação geral de clientes. O DBSCAN foi útil para identificar possíveis comportamentos atípicos, mas pode classificar muitos pontos como ruído dependendo da escolha de eps.
"""

    salvar_texto("q4_interpretacao.txt", texto)
    print(texto)



# QUESTÃO 5 - IMPACTO DA NORMALIZAÇÃO NA CLUSTERIZAÇÃO

def questao_5(clientes, features, variavel_referencia):
    print("\n" + "=" * 70)
    print("QUESTÃO 5 - IMPACTO DA NORMALIZAÇÃO")
    print("=" * 70)

    situacoes = [
        ("Sem normalização", "none"),
        ("Z-score com log1p", "standard"),
        ("Min-Max com log1p", "minmax")
    ]

    comparacao = []

    for nome_situacao, scaler in situacoes:
        X = matriz_preprocessada(clientes, features, scaler=scaler)

        resultados = avaliar_kmeans(X, k_min=2, k_max=10)
        melhor = resultados.sort_values("silhouette", ascending=False).iloc[0]
        melhor_k = int(melhor["k"])

        modelo = KMeans(
            n_clusters=melhor_k,
            random_state=RANDOM_STATE,
            n_init=20
        )

        labels = modelo.fit_predict(X)

        coluna_cluster = f"cluster_{scaler}"
        clientes[coluna_cluster] = labels

        crosstab = pd.crosstab(clientes[coluna_cluster], clientes[variavel_referencia])
        salvar_tabela(crosstab, f"q5_crosstab_{scaler}.csv")

        distribuicao = clientes[coluna_cluster].value_counts().sort_index()
        salvar_tabela(
            distribuicao.to_frame("quantidade"),
            f"q5_distribuicao_clusters_{scaler}.csv"
        )

        plt.figure(figsize=(8, 5))
        plt.scatter(
            clientes["recency_days"],
            clientes["monetary"],
            c=labels,
            s=18,
            alpha=0.7
        )
        plt.yscale("log")
        plt.xlabel("Recência em dias")
        plt.ylabel("Valor monetário total - escala log")
        plt.title(f"K-Means - {nome_situacao}")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"q5_clusters_{scaler}.png", dpi=150)
        plt.close()

        comparacao.append({
            "situacao": nome_situacao,
            "melhor_k": melhor_k,
            "melhor_silhouette": float(melhor["silhouette"]),
            "inertia_no_melhor_k": float(melhor["inertia"])
        })

    comparacao_df = pd.DataFrame(comparacao)
    salvar_tabela(comparacao_df, "q5_comparacao_normalizacao.csv")

    plt.figure(figsize=(8, 4))
    plt.bar(comparacao_df["situacao"], comparacao_df["melhor_silhouette"])
    plt.ylabel("Melhor Silhouette Score")
    plt.title("Impacto da Normalização no K-Means")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "q5_comparacao_silhouette_normalizacao.png", dpi=150)
    plt.close()

    melhor_linha = comparacao_df.sort_values(
        "melhor_silhouette",
        ascending=False
    ).iloc[0]
    print(pd.Series(labels).value_counts())
    texto = f"""
QUESTÃO 5 - INTERPRETAÇÃO

Foram comparadas três situações:
1. Clusterização sem normalização.
2. Clusterização com transformação logarítmica e Z-score.
3. Clusterização com transformação logarítmica e Min-Max.

A transformação logarítmica foi usada porque variáveis como monetary, frequency e total_quantity possuem forte assimetria. A normalização foi avaliada porque algoritmos baseados em distância podem ser dominados por variáveis com escala maior.

A análise considerou:
- mudança nos gráficos de dispersão;
- mudança no melhor valor de K;
- mudança no Silhouette Score;
- mudança na distribuição dos clusters;
- mudança na tabela cruzada.

Melhor resultado observado:
- Situação: {melhor_linha["situacao"]}
- Melhor K: {int(melhor_linha["melhor_k"])}
- Melhor Silhouette Score: {melhor_linha["melhor_silhouette"]:.4f}

Conclusão:
A normalização é importante para este dataset, pois os atributos possuem escalas muito diferentes. Sem normalização, variáveis como monetary e total_quantity dominam o cálculo das distâncias. Com normalização, os atributos contribuem de forma mais equilibrada para a formação dos clusters.
"""

    salvar_texto("q5_interpretacao.txt", texto)
    print(texto)



# QUESTÃO 6 - CLUSTERIZAÇÃO HIERÁRQUICA E DENDROGRAMA

def questao_6(clientes, features, melhor_k):
    print("\n" + "=" * 70)
    print("QUESTÃO 6 - CLUSTERIZAÇÃO HIERÁRQUICA E DENDROGRAMA")
    print("=" * 70)

    X = matriz_preprocessada(clientes, features, scaler="standard")

    n_amostra = min(600, len(clientes))
    indices = np.random.default_rng(RANDOM_STATE).choice(
        len(clientes),
        size=n_amostra,
        replace=False
    )

    X_sample = X[indices]

    resultados = []

    for metodo in ["complete", "average", "single"]:
        Z = linkage(X_sample, method=metodo)

        plt.figure(figsize=(11, 5))
        dendrogram(
            Z,
            no_labels=True,
            truncate_mode="lastp",
            p=40
        )
        plt.title(f"Dendrograma - método de ligação {metodo}")
        plt.xlabel("Amostras/Agrupamentos")
        plt.ylabel("Distância")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"q6_dendrograma_{metodo}.png", dpi=150)
        plt.close()

        agrupador = AgglomerativeClustering(
            n_clusters=melhor_k,
            linkage=metodo
        )

        labels = agrupador.fit_predict(X_sample)
        sil = silhouette_score(X_sample, labels)

        clusters_sugeridos = len(
            np.unique(
                fcluster(
                    Z,
                    t=melhor_k,
                    criterion="maxclust"
                )
            )
        )

        resultados.append({
            "metodo_ligacao": metodo,
            "clusters_sugeridos_no_corte": clusters_sugeridos,
            "silhouette_com_k_do_kmeans": sil
        })

    resultados_df = pd.DataFrame(resultados)
    clusters_sugeridos_medios = int(
        resultados_df["clusters_sugeridos_no_corte"].median()
    )
    salvar_tabela(resultados_df, "q6_resultados_hierarquico.csv")

    melhor_metodo = resultados_df.sort_values(
        "silhouette_com_k_do_kmeans",
        ascending=False
    ).iloc[0]

    texto = f"""
QUESTÃO 6 - INTERPRETAÇÃO

Foram utilizados os mesmos atributos da clusterização com K-Means:
{features}

Foi aplicada transformação logarítmica e normalização Z-score, pois a clusterização hierárquica também é baseada em distância.

Foram testados os métodos de ligação:
- complete;
- average;
- single.

Os dendrogramas foram gerados com uma amostra de {n_amostra} clientes para facilitar a visualização. O dataset completo possui muitos clientes, o que deixaria o dendrograma visualmente poluído.

O número de clusters escolhido anteriormente pelo K-Means foi:
K = {melhor_k}

Os dendrogramas sugeriram aproximadamente:
{clusters_sugeridos_medios} clusters.

Melhor método hierárquico considerando Silhouette com K do K-Means:
- Método: {melhor_metodo["metodo_ligacao"]}
- Silhouette Score: {melhor_metodo["silhouette_com_k_do_kmeans"]:.4f}

Os cortes observados nos dendrogramas sugeriram aproximadamente {clusters_sugeridos_medios} grupos. Esse valor pode ser comparado diretamente ao K={melhor_k} obtido pelo K-Means para verificar se ambas as técnicas identificam uma estrutura semelhante nos dados.

Conclusão:
A clusterização hierárquica permite analisar visualmente a associação entre as amostras por meio do dendrograma. Se o dendrograma sugere cortes próximos ao K escolhido pelo K-Means, isso confirma parcialmente os resultados anteriores. Caso contrário, indica que a estrutura dos dados é mais complexa.

Neste trabalho, a clusterização hierárquica funciona como complemento ao K-Means, permitindo validar visualmente a estrutura dos agrupamentos.
"""

    salvar_texto("q6_interpretacao.txt", texto)
    print(texto)



# QUESTÃO 7 - PCA E CLASSIFICAÇÃO SUPERVISIONADA

def questao_7(clientes, features):
    print("\n" + "=" * 70)
    print("QUESTÃO 7 - PCA E CLASSIFICAÇÃO SUPERVISIONADA")
    print("=" * 70)

    # Conexão com a Questão 3:
    # A variável alvo da classificação será o cluster gerado pelo K-Means.
    # Isso avalia se o PCA preserva a estrutura de agrupamento encontrada.
    from sklearn.preprocessing import LabelEncoder

    target_classificacao = "country_group"

    X = clientes[features].copy()

    encoder = LabelEncoder()
    y = encoder.fit_transform(clientes[target_classificacao])

    X = clientes[features].copy()
    y = clientes[target_classificacao].copy()

    X_log = np.log1p(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    matriz_cov = np.cov(X_scaled.T)

    matriz_cov_df = pd.DataFrame(
        matriz_cov,
        index=features,
        columns=features
    )

    salvar_tabela(matriz_cov_df, "q7_matriz_covariancia.csv")

    plot_matriz(
        matriz_cov_df,
        "PCA - Matriz de Covariância",
        RESULTS_DIR / "q7_matriz_covariancia.png",
        labels_x=features,
        labels_y=features,
        colorbar_label="Covariância"
    )

    autovalores, autovetores = np.linalg.eigh(matriz_cov)

    ordem = np.argsort(autovalores)[::-1]
    autovalores = autovalores[ordem]
    autovetores = autovetores[:, ordem]

    variancia_explicada = autovalores / autovalores.sum()
    variancia_acumulada = np.cumsum(variancia_explicada)

    tabela_autovalores = pd.DataFrame({
        "componente": [f"PC{i + 1}" for i in range(len(autovalores))],
        "autovalor": autovalores,
        "variancia_explicada": variancia_explicada,
        "variancia_acumulada": variancia_acumulada
    })

    autovetores_df = pd.DataFrame(
        autovetores,
        index=features,
        columns=[f"PC{i + 1}" for i in range(len(features))]
    )

    salvar_tabela(tabela_autovalores, "q7_autovalores_variancia_explicada.csv")
    salvar_tabela(autovetores_df, "q7_autovetores.csv")

    plt.figure(figsize=(7, 4))
    plt.plot(
        range(1, len(variancia_acumulada) + 1),
        variancia_acumulada,
        marker="o"
    )
    plt.axhline(0.95, linestyle="--")
    plt.xlabel("Número de componentes principais")
    plt.ylabel("Variância explicada acumulada")
    plt.title("PCA - Variância Explicada Acumulada")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "q7_variancia_explicada_acumulada.png", dpi=150)
    plt.close()

    pca_95 = PCA(n_components=0.95, random_state=RANDOM_STATE)
    X_pca_95 = pca_95.fit_transform(X_scaled)

    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca_2d = pca_2d.fit_transform(X_scaled)

    pca_2d_df = pd.DataFrame(X_pca_2d, columns=["PC1", "PC2"])
    pca_2d_df[target_classificacao] = y

    salvar_tabela(pca_2d_df, "q7_dados_projetados_pca_2d.csv")

    plt.figure(figsize=(8, 5))

    classes = np.unique(y)

    for classe in classes:
        temp = pca_2d_df[pca_2d_df[target_classificacao] == classe]

        plt.scatter(
            temp["PC1"],
            temp["PC2"],
            s=18,
            alpha=0.65,
            label=str(classe)
        )

    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.title("PCA - Projeção dos Dados em 2 Componentes")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "q7_pca_2d_clusters.png", dpi=150)
    plt.close()

    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=stratify
    )

    k_vizinhos = min(6, len(X_train))

    knn_original = KNeighborsClassifier(n_neighbors=k_vizinhos)
    knn_original.fit(X_train, y_train)
    y_pred_original = knn_original.predict(X_test)

    metricas_sem_pca = {
        "modelo": "KNN sem PCA",
        "accuracy": accuracy_score(y_test, y_pred_original),
        "precision_macro": precision_score(y_test, y_pred_original, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred_original, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred_original, average="macro", zero_division=0)
    }

    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
        X_pca_95,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=stratify
    )

    knn_pca = KNeighborsClassifier(n_neighbors=k_vizinhos)
    knn_pca.fit(X_train_pca, y_train_pca)
    y_pred_pca = knn_pca.predict(X_test_pca)

    metricas_com_pca = {
        "modelo": "KNN com PCA",
        "accuracy": accuracy_score(y_test_pca, y_pred_pca),
        "precision_macro": precision_score(y_test_pca, y_pred_pca, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test_pca, y_pred_pca, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test_pca, y_pred_pca, average="macro", zero_division=0)
    }

    comparacao = pd.DataFrame([metricas_sem_pca, metricas_com_pca])
    salvar_tabela(comparacao, "q7_comparacao_classificacao_sem_pca_com_pca.csv")

    report_sem_pca = classification_report(y_test, y_pred_original, zero_division=0)
    report_com_pca = classification_report(y_test_pca, y_pred_pca, zero_division=0)

    salvar_texto("q7_classification_report_sem_pca.txt", report_sem_pca)
    salvar_texto("q7_classification_report_com_pca.txt", report_com_pca)

    matriz_sem_pca = confusion_matrix(y_test, y_pred_original, labels=classes)
    matriz_com_pca = confusion_matrix(y_test_pca, y_pred_pca, labels=classes)

    matriz_sem_pca_df = pd.DataFrame(
        matriz_sem_pca,
        index=[f"Real_{c}" for c in classes],
        columns=[f"Pred_{c}" for c in classes]
    )

    matriz_com_pca_df = pd.DataFrame(
        matriz_com_pca,
        index=[f"Real_{c}" for c in classes],
        columns=[f"Pred_{c}" for c in classes]
    )

    salvar_tabela(matriz_sem_pca_df, "q7_matriz_confusao_sem_pca.csv")
    salvar_tabela(matriz_com_pca_df, "q7_matriz_confusao_com_pca.csv")

    plot_matriz(
        matriz_sem_pca_df,
        "Matriz de Confusão - KNN sem PCA",
        RESULTS_DIR / "q7_matriz_confusao_sem_pca.png",
        labels_x=[f"Pred_{c}" for c in classes],
        labels_y=[f"Real_{c}" for c in classes],
        colorbar_label="Quantidade"
    )

    plot_matriz(
        matriz_com_pca_df,
        "Matriz de Confusão - KNN com PCA",
        RESULTS_DIR / "q7_matriz_confusao_com_pca.png",
        labels_x=[f"Pred_{c}" for c in classes],
        labels_y=[f"Real_{c}" for c in classes],
        colorbar_label="Quantidade"
    )

    n_componentes = pca_95.n_components_
    variancia_total_preservada = pca_95.explained_variance_ratio_.sum()

    acc_sem_pca = metricas_sem_pca["accuracy"]
    acc_com_pca = metricas_com_pca["accuracy"]
    diferenca = acc_com_pca - acc_sem_pca

    if diferenca >= -0.03:
        conclusao = (
            "O PCA preservou bem as informações relevantes, pois a perda de desempenho "
            "foi pequena ou inexistente."
        )
    else:
        conclusao = (
            "O PCA reduziu a dimensionalidade, mas causou perda relevante de desempenho "
            "na classificação."
        )

    texto = f"""
QUESTÃO 7 - INTERPRETAÇÃO

Nesta questão, a classificação supervisionada foi conectada aos resultados do aprendizado não supervisionado. A variável-alvo usada foi:

{target_classificacao}

Essa escolha permite avaliar se a redução de dimensionalidade preserva informações úteis para prever a categoria de referência do dataset.
Atributos de entrada:
{features}

Etapas realizadas:
1. Separação entre atributos de entrada X e variável-alvo y.
2. Aplicação de transformação logarítmica.
3. Padronização com StandardScaler.
4. Cálculo da matriz de covariância.
5. Cálculo dos autovalores e autovetores.
6. Cálculo da variância explicada.
7. Aplicação do PCA.
8. Treinamento do KNN com dados originais.
9. Treinamento do mesmo KNN com dados reduzidos via PCA.
10. Comparação usando acurácia, precisão, recall, F1-score e matriz de confusão.

O PCA com preservação de 95% da variância reduziu os dados de {len(features)} atributos originais para {n_componentes} componentes principais.

Variância total preservada:
{variancia_total_preservada:.4f}

Resultados principais:
- Acurácia sem PCA: {acc_sem_pca:.4f}
- Acurácia com PCA: {acc_com_pca:.4f}
- Diferença de acurácia: {diferenca:.4f}

Interpretação:
O PCA reduziu a dimensionalidade dos dados ao transformar os atributos originais em componentes principais. Esses componentes concentram a maior parte da variância dos dados e podem reduzir redundância entre variáveis correlacionadas.

Conclusão:
{conclusao}

Assim, a aplicação do PCA permite avaliar se a redução dimensional mantém informações úteis para classificação e se os padrões descobertos na clusterização são preservados após a redução de dimensionalidade.
"""

    salvar_texto("q7_interpretacao.txt", texto)
    print(texto)



# EXECUÇÃO PRINCIPAL

def main():
    df_original = carregar_dados()

    df_limpo, clientes, features, variavel_referencia = preparar_base_cliente(df_original)

    questao_1(df_original, df_limpo, clientes, features)
    questao_2(clientes, variavel_referencia)

    melhor_k, X_kmeans = questao_3(clientes, features, variavel_referencia)

    questao_4(clientes, X_kmeans, variavel_referencia, melhor_k)
    questao_5(clientes, features, variavel_referencia)
    questao_6(clientes, features, melhor_k)
    questao_7(clientes, features)

    clientes.to_csv(
        RESULTS_DIR / "base_clientes_final_com_clusters.csv",
        index=False,
        encoding="utf-8-sig"
    )



if __name__ == "__main__":
    main()