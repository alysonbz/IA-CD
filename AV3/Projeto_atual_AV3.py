# ============================================================
# TRABALHO COMPUTACIONAL - AV3
# Dataset: Bank Marketing
# ============================================================

# ============================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import numpy as np

# ==============================================================================
# CONFIGURAÇÃO INICIAL E LEITURA DOS DADOS
# ==============================================================================
df = pd.read_csv('bank.csv', sep=';')

# ==============================================================================
# QUESTÃO 1 – ANÁLISE EXPLORATÓRIA DOS DADOS (EDA)
# ==============================================================================
print("\n" + "="*60)
print("QUESTÃO 1: ANÁLISE EXPLORATÓRIA")
print("="*60)

# ---------------------------------
#  VERIFICAÇÃO GERAL DO DATASET
# ---------------------------------
print(f"\nQuantidade de linhas: {df.shape[0]}")
print(f"Quantidade de colunas: {df.shape[1]}")

print("\nTipos das variáveis:")
print(df.dtypes)

print("\nPrimeiras observações:")
print(df.head())

# ---------------------------------
# VERIFICAÇÃO DE PROBLEMAS
# ---------------------------------
print("\nValores ausentes:")
print(df.isnull().sum())

print("\nTotal de valores ausentes:")
print(df.isnull().sum().sum())

print("\nRegistros duplicados:")
print(df.duplicated().sum())

# ---------------------------------
# ESTATÍSTICAS DESCRITIVAS
# ---------------------------------
print("ESTATÍSTICAS DESCRITIVAS")
pd.set_option('display.max_columns', None)   # 1. Força o Pandas a NUNCA esconder colunas do relatório
pd.set_option('display.width', 1000)         # 2. Impede que o texto quebre e suma com colunas na tela do terminal
print(df.describe(include=['number']).T)     # 3. Mostrar apenas as colunas numéricas transposta

# -----------------------------------
# VARIÁVEIS NUMÉRICAS E CATEGÓRICAS
# -----------------------------------
numericas = df.select_dtypes(include=np.number).columns
categoricas = df.select_dtypes(include=['object', 'string']).columns

print("\nVariáveis numéricas:")
print(list(numericas))

print("\nVariáveis categóricas:")
print(list(categoricas))

# ----------------------------------------------------
# EXIBINDO HISTOGRAMA DAS VARIÁVEIS NUMÉRICAS
# ----------------------------------------------------
plt.figure(figsize=(12, 8))
for i, col in enumerate(['age', 'balance', 'duration', 'campaign', 'pdays', 'previous'], 1):
    plt.subplot(3, 2, i)
    sns.histplot(df[col], bins=30, color='skyblue', stat='density')
    sns.kdeplot(df[col], color='red', linewidth=1.5)
    plt.title(f'Distribuição de {col.capitalize()}')
plt.tight_layout()
plt.show()

# --------------------------------------------------
# EXIBINDO O GRÁFICO DE BARRAS DA VARIÁVEL ALVO
# --------------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='y', hue='y', palette='Set2', legend=False)
plt.title('Distribuição da Variável Alvo (y)')
plt.xlabel('Aceitou o depósito?')
plt.ylabel('Contagem')
plt.tight_layout()
plt.show()

# --------------------------------------------
# MATRIZ DE CORRELAÇÃO - VARIÁVEIS NUMÉRICAS
# --------------------------------------------
corr = df[numericas].corr()
print("\n" + "="*60)
print("MATRIZ DE CORRELAÇÃO")
print("="*60)
print(corr)


plt.figure(figsize=(10,8))
sns.heatmap( corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlação')
plt.show()

# ==============================================================================
# QUESTÃO 2 – ANÁLISE VISUAL COM DOIS ATRIBUTOS
# ==============================================================================
print("\n--- QUESTÃO 2: ANÁLISE VISUAL COM DOIS ATRIBUTOS ---")

# ---------------------------------
# GRÁFICO 1 - AGE x BALANCE
# ---------------------------------
plt.figure(figsize=(10,6))
sns.scatterplot( data=df, x='age', y='balance', hue='y', alpha=0.7)
plt.title('Análise Visual de Clusters: Age x Balance')
plt.xlabel('Idade')
plt.ylabel('Saldo')
plt.show()

# ---------------------------------------------------------
# BALANCE x DURATION - COM E SEM VALORES EXTREMOS
# ---------------------------------------------------------

q1 = df['balance'].quantile(0.01)
q99 = df['balance'].quantile(0.99)

fig, axes = plt.subplots(
    1, 2,
    figsize=(16,6)
)

# ------------------------------------------------------------
# COM EXTREMOS
# ------------------------------------------------------------

sns.scatterplot( data=df, x='balance', y='duration', hue='y', alpha=0.7, ax=axes[0])
axes[0].set_title('Balance x Duration (Com extremos)')
axes[0].set_xlabel('Saldo')
axes[0].set_ylabel('Duração do contato')

# ------------------------------------------------------------
# SEM EXTREMOS
# ------------------------------------------------------------

sns.scatterplot( data=df, x='balance', y='duration', hue='y', alpha=0.7, ax=axes[1])
axes[1].set_xlim(q1, q99)
axes[1].set_title('Balance x Duration (Sem extremos)')
axes[1].set_xlabel('Saldo')
axes[1].set_ylabel('Duração do contato')

# Remover legenda duplicada
axes[1].legend_.remove()
plt.suptitle('Comparação: Balance x Duration', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ==============================================================================
# QUESTÃO 3 – CLUSTERIZAÇÃO COM K-MEANS AND ESCOLHA DO K
# ==============================================================================
print("\n" + "="*60)
print("QUESTÃO 3: K-MEANS")
print("="*60)

# ------------------------------------------------------------------------------
# SELEÇÃO DOS ATRIBUTOS
# ------------------------------------------------------------------------------

atributos_cluster = [ 'age', 'balance', 'duration', 'campaign']
X = df[atributos_cluster]

print("\nAtributos utilizados:")
print(atributos_cluster)

# ------------------------------------------------------------------------------
# NORMALIZAÇÃO DOS DADOS
# ------------------------------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------------------------
# MÉTODO DO COTOVELO
# ------------------------------------------------------------------------------

inercia = []

valores_k = range(2, 11)

for k in valores_k:

    modelo = KMeans( n_clusters=k, random_state=42, n_init=10    )
    modelo.fit(X_scaled)
    inercia.append(modelo.inertia_)

# ------------------------------------------------------------------------------
# SILHOUETTE SCORE
# ------------------------------------------------------------------------------

valores_k = list(range(2, 11))
silhouettes = []

for k in valores_k:
    modelo = KMeans( n_clusters=k, random_state=42, n_init=10)
    labels = modelo.fit_predict(X_scaled)
    score = silhouette_score( X_scaled, labels    )

    silhouettes.append(score)

# ------------------------------------------------------------------------------
# GRÁFICO COMPARATIVO: COTOVELO VS SILHUETA
# ------------------------------------------------------------------------------

fig, ax1 = plt.subplots(figsize=(10, 6))

# Configuração do primeiro eixo (Inércia - Gráfico de Linha Vermelha)
cor_inercia = 'tab:red'
ax1.set_xlabel('Número de Clusters (K)', fontweight='bold')
ax1.set_ylabel('Inércia (Método do Cotovelo)', color=cor_inercia, fontweight='bold')
linha1 = ax1.plot(valores_k, inercia, marker='o', color=cor_inercia, linewidth=2, label='Inércia')
ax1.tick_params(axis='y', labelcolor=cor_inercia)
ax1.grid(True, linestyle='--', alpha=0.5)

# Criando um segundo eixo Y que compartilha o mesmo eixo X
ax2 = ax1.twinx()

# Configuração do segundo eixo (Silhouette Score - Gráfico de Linha Azul)
cor_silhueta = 'tab:blue'
ax2.set_ylabel('Silhouette Score (Silhueta)', color=cor_silhueta, fontweight='bold')
linha2 = ax2.plot(valores_k, silhouettes, marker='s', color=cor_silhueta, linewidth=2, linestyle='--', label='Silhueta')
ax2.tick_params(axis='y', labelcolor=cor_silhueta)

# Ajustes de legenda unificada e título
linhas = linha1 + linha2
legendas = [l.get_label() for l in linhas]
ax1.legend(linhas, legendas, loc='upper right')

plt.title('Análise Comparativa: Inércia vs. Silhouette Score por valor de K', fontsize=14, fontweight='bold', pad=15)
fig.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# TABELA COMPARATIVA NO CONSOLE
# ------------------------------------------------------------------------------
df_comparativo = pd.DataFrame({
    'K (Clusters)': valores_k,
    'Inércia (Cotovelo)': inercia,
    'Silhouette Score': silhouettes
})

print("\n" + "="*50)
print("TABELA COMPARATIVA DE MÉTRICAS")
print("="*50)
print(df_comparativo.to_string(index=False, formatters={'Inércia (Cotovelo)': '{:,.2f}'.format, 'Silhouette Score': '{:.4f}'.format}))

# ------------------------------------------------------------------------------
# COMPARAÇÃO VISUAL DOS CLUSTERS: COTOVELO (K=6) VS SILHUETA
# ------------------------------------------------------------------------------

# 1. Definindo os dois melhores K identificados pelos critérios
k_silhueta = valores_k[np.argmax(silhouettes)]
k_cotovelo = 6

print(f"\nGerando gráficos comparativos...")
print(f"-> Cenário A (Melhor Silhueta): K = {k_silhueta}")
print(f"-> Cenário B (Melhor Cotovelo): K = {k_cotovelo}")

# Escolha do melhor K
print("\nCONCLUSÃO:")

melhor_k = k_silhueta

print( f"K escolhido para o modelo final: {melhor_k} "
    f"(maior Silhouette Score)")

kmeans_final = KMeans( n_clusters=melhor_k, random_state=42, n_init=10)
df['cluster_final'] = kmeans_final.fit_predict(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot( data=df, x='duration', y='balance', hue='cluster_final', palette='viridis', alpha=0.7)
plt.title(f'Clusters Finais do K-Means (K={melhor_k})')
plt.show()

centroides = pd.DataFrame( scaler.inverse_transform(kmeans_final.cluster_centers_),
    columns=atributos_cluster)
print(f'\nInterpretação dos Cluters')
print(centroides.round(2))

# ------------------------------------------------------------------------------
# TABULAÇÃO CRUZADA: CLUSTERS VS ADESÃO AO PRODUTO (COLUNA Y)
# ------------------------------------------------------------------------------
print(f"\nANÁLISE DE CONVERSÃO POR CLUSTER (MODELO FINAL K={melhor_k})")

# 1. Tabela com a contagem absoluta de clientes
crosstab_absoluto = pd.crosstab(
    df['cluster_final'],  # <-- Nome corrigido da coluna
    df['y'])
print("\nQuantidade absoluta de clientes por grupo:")
print(crosstab_absoluto)

# 2. Tabela com as porcentagens (Taxa de conversão por linha)
crosstab_porcentagem = pd.crosstab(
    df['cluster_final'], df['y'], normalize='index') * 100
print("\nPercentual (%) dentro de cada grupo:")
# O '.round(2)' limita as casas decimais para facilitar a leitura no console
print(crosstab_porcentagem.round(2))

# ==============================================================================
# QUESTÃO 4 – COMPARAÇÃO ENTRE K-MEANS E DBSCAN
# ==============================================================================
print("\n" + "="*60)
print("\n--- QUESTÃO 4: COMPARAÇÃO ENTRE K-MEANS E DBSCAN ---")
print("="*60)
min_samples_definido = 8

vizinhos = NearestNeighbors(n_neighbors=min_samples_definido)
vizinhos_ajustados = vizinhos.fit(X_scaled)
distancias, indices = vizinhos_ajustados.kneighbors(X_scaled)
distancias_ordenadas = np.sort(distancias[:, min_samples_definido - 1], axis=0)

# Gráfico de K-Distância (Heurística EPS)
plt.figure(figsize=(8, 5))
plt.plot(distancias_ordenadas, color='darkorange', linewidth=2)
plt.title('Gráfico de K-Distância (Encontrar o Joelho do DBSCAN)')
plt.xlabel('Pontos ordenados')
plt.ylabel('Distância')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Rodando o DBSCAN
dbscan = DBSCAN(eps=0.6, min_samples=min_samples_definido)
df['cluster_dbscan'] = dbscan.fit_predict(X_scaled)

# Gráfico dos Clusters Gerados pelo DBSCAN
plt.figure(figsize=(8,5))

# Cluster principal
sns.scatterplot( data=df[df['cluster_dbscan']==0], x='duration', y='balance', color='royalblue',
    s=40, alpha=0.5, label='Cluster 0')

# Cluster secundário
sns.scatterplot( data=df[df['cluster_dbscan']==1], x='duration', y='balance', color='red',
    s=80, alpha=1, label='Cluster 1')

# Ruídos
sns.scatterplot( data=df[df['cluster_dbscan']==-1], x='duration', y='balance', color='gray',
    s=50, alpha=0.7, label='Ruído -1')

plt.title('Agrupamentos Gerados pelo DBSCAN (Ruídos em Cinza/Negativo)')
plt.xlabel('Duração da Contato (duration)')
plt.ylabel('Saldo Bancário (balance)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ----------------------------------------------------
# ESTATÍSTICAS DO DBSCAN
# ----------------------------------------------------
labels = df['cluster_dbscan']

n_clusters = len(set(labels)) - (1 if -1 in labels.values else 0)

n_ruidos = np.sum(labels == -1)

perc_ruidos = (n_ruidos / len(df)) * 100

print("\nRESULTADOS DO DBSCAN")
print(f"Clusters encontrados: {n_clusters}")
print(f"Ruídos encontrados: {n_ruidos}")
print(f"Percentual de ruídos: {perc_ruidos:.2f}%")

# ==============================================================================
# QUESTÃO 5 – IMPACTO DA NORMALIZAÇÃO NA CLUSTERIZAÇÃO
# ==============================================================================
print("\n" + "="*60)
print("--- QUESTÃO 5: IMPACTO DA NORMALIZAÇÃO ---")
print("="*60)

X_raw = df[['age', 'balance', 'duration', 'campaign']].values

# FUNÇÃO PARA ENCONTRAR MELHOR K PELO SILHOUETTE

def melhor_k_silhouette(X, k_min=2, k_max=10):

    melhores_scores = []

    for k in range(k_min, k_max + 1):
        modelo = KMeans( n_clusters=k, random_state=42, n_init=10)
        labels = modelo.fit_predict(X)
        score = silhouette_score(X, labels)
        melhores_scores.append(score)
    melhor_k = np.argmax(melhores_scores) + k_min
    return melhor_k, max(melhores_scores), melhores_scores

# Dados originais
k_raw, sil_raw, scores_raw = melhor_k_silhouette(X_raw)

# Z-score
scaler_zscore = StandardScaler()
X_zscore = scaler_zscore.fit_transform(X_raw)
k_zscore, sil_zscore, scores_zscore = melhor_k_silhouette(X_zscore)

# Dados Min-Max
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X_raw)
k_minmax, sil_minmax, scores_minmax = melhor_k_silhouette(X_minmax)

# -------------------------------------------------------
# GRÁFICO DO SILHOUETTE SCORE
# -------------------------------------------------------
valores_k = range(2, 11)

plt.figure(figsize=(10,6))

plt.plot(
    valores_k,
    scores_raw,
    marker='o',
    label='Sem Normalização'
)

plt.plot(
    valores_k,
    scores_zscore,
    marker='o',
    label='Z-Score'
)

plt.plot(
    valores_k,
    scores_minmax,
    marker='o',
    label='Min-Max'
)

plt.xlabel('Número de Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Comparação dos Silhouette Scores')
plt.xticks(valores_k)
plt.grid(True)
plt.legend()

plt.show()

print("\nMELHOR K EM CADA CENÁRIO")
print(f"Sem normalização -> K={k_raw} | Silhouette={sil_raw:.4f}")
print(f"Z-score          -> K={k_zscore} | Silhouette={sil_zscore:.4f}")
print(f"Min-Max          -> K={k_minmax} | Silhouette={sil_minmax:.4f}")

# Sem normalização
kmeans_raw = KMeans(n_clusters=k_raw,random_state=42, n_init=10)
labels_raw = kmeans_raw.fit_predict(X_raw)

# Z-score
kmeans_zscore = KMeans( n_clusters=k_zscore, random_state=42, n_init=10)
labels_zscore = kmeans_zscore.fit_predict(X_zscore)

# Min-Max
kmeans_minmax = KMeans(n_clusters=k_minmax,random_state=42,n_init=10)
labels_minmax = kmeans_minmax.fit_predict(X_minmax)

# Distribuição  dos Clusters
print("\nDISTRIBUIÇÃO DOS CLUSTERS")

print("\nSem Normalização")
print(pd.Series(labels_raw).value_counts().sort_index())

print("\nZ-score")
print(pd.Series(labels_zscore).value_counts().sort_index())

print("\nMin-Max")
print(pd.Series(labels_minmax).value_counts().sort_index())

# Tabela crusada Crosstab
print("\nTABELAS CRUZADAS (%)")

print("\nSem Normalização")
print( pd.crosstab( labels_raw, df['y'], normalize='index').round(4) * 100)

print("\nZ-score")
print(pd.crosstab( labels_zscore, df['y'], normalize='index' ).round(4) * 100)

print("\nMin-Max")
print(pd.crosstab( labels_minmax, df['y'], normalize='index').round(4) * 100)

resultado_norm = pd.DataFrame({
    'Cenário': ['Sem Normalização','Z-score','Min-Max' ],
    'Melhor K': [ k_raw, k_zscore, k_minmax ],
    'Silhouette Score': [ sil_raw, sil_zscore, sil_minmax ]})

print("\nCOMPARAÇÃO DAS NORMALIZAÇÕES")
print(resultado_norm.round(4))

# ----------------------------------------------------
# GRÁFICO DE COMPARAÇÃO
# ----------------------------------------------------
plt.figure(figsize=(18,5))

# Sem normalização
plt.subplot(1,3,1)
sns.scatterplot( x=df['duration'], y=df['balance'], hue=labels_raw, palette='Set1', alpha=0.6)
plt.title(f'Sem Normalização (K={k_raw})')
plt.xlabel('Duration')
plt.ylabel('Balance')

# Z-score
plt.subplot(1,3,2)
sns.scatterplot( x=df['duration'], y=df['balance'], hue=labels_zscore, palette='Set1', alpha=0.6)
plt.title(f'Z-score (K={k_zscore})')
plt.xlabel('Duration')
plt.ylabel('Balance')

# Min-Max
plt.subplot(1,3,3)
sns.scatterplot( x=df['duration'], y=df['balance'], hue=labels_minmax, palette='Set1', alpha=0.6)
plt.title(f'Min-Max (K={k_minmax})')
plt.xlabel('Duration')
plt.ylabel('Balance')

plt.tight_layout()
plt.show()

# ==============================================================================
# QUESTÃO 6 – CLUSTERIZAÇÃO HIERÁRQUICA E DENDROGRAMA
# ==============================================================================
print("\n" + "="*60)
print("QUESTÃO 6 - CLUSTERIZAÇÃO HIERÁRQUICA")
print("="*60)

# ------------------------------------------------------------------------------
# 1. SELEÇÃO DOS ATRIBUTOS
# ------------------------------------------------------------------------------

atributos = ['age', 'balance', 'duration', 'campaign']
X = df[atributos]

print("\nAtributos utilizados:")
print(atributos)

# ------------------------------------------------------------------------------
# 2. NORMALIZAÇÃO DOS DADOS
# ------------------------------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nDados normalizados utilizando StandardScaler (Z-score).")

# ------------------------------------------------------------------------------
# 3. AMOSTRAGEM PARA O DENDROGRAMA
# ------------------------------------------------------------------------------

amostra = X.sample( n=300, random_state=42)
X_amostra = StandardScaler().fit_transform(amostra)

print("Amostra utilizada para os dendrogramas: 300 registros")

# ------------------------------------------------------------------------------
# FUNÇÃO AUXILIAR
# ------------------------------------------------------------------------------

def analisar_metodo(nome_metodo, metodo_linkage, corte):

    print("\n" + "-"*60)
    print(f"MÉTODO {nome_metodo.upper()}")
    print("-"*60)

    # Matriz de ligação
    matriz_linkage = linkage(X_amostra, method=metodo_linkage)

    # Dendrograma
    plt.figure(figsize=(12,6))
    dendrogram( matriz_linkage, truncate_mode='level', p=5 )

    plt.axhline( y=corte, color='red', linestyle='--', label=f'Corte = {corte}')

    plt.title(f"Dendrograma - {nome_metodo}")
    plt.xlabel("Amostras")
    plt.ylabel("Distância")
    plt.legend()

    plt.show()

    # Clusterização a partir do corte
    clusters = fcluster( matriz_linkage, t=corte, criterion='distance')
    qtd_clusters = len(np.unique(clusters))
    sil = silhouette_score(X_amostra, clusters)

    print(f"Corte adotado: {corte}")
    print(f"Quantidade de clusters: {qtd_clusters}")
    print(f"Silhouette Score: {sil:.4f}")

    return qtd_clusters, sil


# ==============================================================================
# SINGLE LINKAGE
# ==============================================================================

qtd_single, sil_single = analisar_metodo(
    nome_metodo='Single Linkage',
    metodo_linkage='single',
    corte=5
)

# ==============================================================================
# AVERAGE LINKAGE
# ==============================================================================

qtd_average, sil_average = analisar_metodo(
    nome_metodo='Average Linkage',
    metodo_linkage='average',
    corte=10
)

# ==============================================================================
# COMPLETE LINKAGE
# ==============================================================================

qtd_complete, sil_complete = analisar_metodo(
    nome_metodo='Complete Linkage',
    metodo_linkage='complete',
    corte=13
)

# ==============================================================================
# TABELA RESUMO
# ==============================================================================

resultado_hierarquico = pd.DataFrame({

    'Método': ['Single', 'Average', 'Complete'],
    'Clusters Encontrados': [qtd_single, qtd_average, qtd_complete],
    'Silhouette Score': [sil_single, sil_average, sil_complete]})

print("\n" + "="*60)
print("RESUMO DOS MÉTODOS HIERÁRQUICOS")
print("="*60)

print(resultado_hierarquico.round(4))

# ==============================================================================
# COMPARAÇÃO COM O K-MEANS
# ==============================================================================

print("\n" + "="*60)
print("COMPARAÇÃO COM O K-MEANS")
print("="*60)

print(f"K-Means (Questão 3): K = {melhor_k}")

print(f"Single Linkage : {qtd_single} clusters")
print(f"Average Linkage: {qtd_average} clusters")
print(f"Complete Linkage: {qtd_complete} clusters")

# ==============================================================================
# INTERPRETAÇÃO AUTOMÁTICA
# ==============================================================================

print("\n" + "="*60)
print("ANÁLISE FINAL")
print("="*60)

if qtd_complete == melhor_k:

    print(
        f"O método Complete sugeriu exatamente "
        f"{melhor_k} clusters, confirmando o resultado "
        f"obtido anteriormente pelo K-Means."  )

elif abs(qtd_complete - melhor_k) <= 1:

    print(
        "O método Complete apresentou uma quantidade "
        "de grupos muito próxima daquela encontrada "
        "pelo K-Means."   )

else:

    print(
        "O método Complete apresentou uma estrutura "
        "de agrupamentos diferente daquela encontrada "
        "pelo K-Means."   )

if qtd_average < qtd_complete:

    print(
        "O método Average produziu uma quantidade "
        "menor de grupos devido à utilização da "
        "distância média entre clusters."  )

if qtd_single < qtd_average:

    print(
        "O método Single apresentou efeito de "
        "encadeamento (chaining), formando grupos "
        "menos compactos e reduzindo o número de "
        "clusters identificados."  )

print(
    "\nA clusterização hierárquica foi utilizada "
    "como uma ferramenta complementar ao K-Means. "
    "A comparação entre os métodos permite verificar "
    "se a estrutura encontrada nos dados é consistente "
    "entre diferentes abordagens de agrupamento.")


# ==============================================================================
# QUESTÃO 7 – REDUÇÃO DE DIMENSIONALIDADE COM PCA E CLASSIFICAÇÃO
# ==============================================================================

print("\n" + "="*60)
print("QUESTÃO 7 – PCA E CLASSIFICAÇÃO SUPERVISIONADA")
print("="*60)

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import time

# ------------------------------------------------------------------------------
# 1. ATRIBUTOS E VARIÁVEL-ALVO
# ------------------------------------------------------------------------------

atributos = [ 'age', 'balance', 'duration', 'campaign']

X = df[atributos]
y = df['y'].map({ 'no': 0, 'yes': 1})

# ------------------------------------------------------------------------------
# 2. NORMALIZAÇÃO (Z-SCORE)
# ------------------------------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------------------------
# 3. DIVISÃO TREINO E TESTE
# ------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------------------------------
# 4. PCA
# ------------------------------------------------------------------------------

pca = PCA(n_components=0.95)

X_train_pca = pca.fit_transform(X_train)

X_test_pca = pca.transform(X_test)

print("\nANÁLISE DO PCA")
print(f"Componentes originais: {X_train.shape[1]}")
print(f"Componentes após PCA: {pca.n_components_}")
print(f"Variância preservada: {sum(pca.explained_variance_ratio_)*100:.2f}%")

# ------------------------------------------------------------------------------
# 5. KNN SEM PCA
# ------------------------------------------------------------------------------

inicio = time.time()

knn_original = KNeighborsClassifier(n_neighbors=5)

knn_original.fit( X_train, y_train)
tempo_knn_original = time.time() - inicio
y_pred_knn_original = knn_original.predict(X_test)

# ------------------------------------------------------------------------------
# 6. KNN COM PCA
# ------------------------------------------------------------------------------

inicio = time.time()

knn_pca = KNeighborsClassifier( n_neighbors=5)

knn_pca.fit( X_train_pca, y_train)
tempo_knn_pca = time.time() - inicio
y_pred_knn_pca = knn_pca.predict(X_test_pca)


# ------------------------------------------------------------------------------
# 9. ACURÁCIAS
# ------------------------------------------------------------------------------

acc_knn_original = accuracy_score( y_test, y_pred_knn_original)
acc_knn_pca = accuracy_score( y_test, y_pred_knn_pca)

# ------------------------------------------------------------------------------
# 10. RELATÓRIOS DE CLASSIFICAÇÃO
# ------------------------------------------------------------------------------

print("\n" + "="*60)
print("KNN - DADOS ORIGINAIS")
print("="*60)

print(classification_report(y_test, y_pred_knn_original, target_names=['no', 'yes']))

print("\n" + "="*60)
print("KNN - COM PCA")
print("="*60)

print(classification_report(y_test, y_pred_knn_pca, target_names=['no', 'yes']))


# ------------------------------------------------------------------------------
# 11. TABELA COMPARATIVA
# ------------------------------------------------------------------------------

resultado_pca = pd.DataFrame({
    'Modelo': ['KNN sem PCA', 'KNN com PCA'],
    'Accuracy': [acc_knn_original, acc_knn_pca],
    'Tempo Treino (s)': [ tempo_knn_original, tempo_knn_pca]
})

print("\n" + "="*60)
print("COMPARAÇÃO DOS MODELOS")
print("="*60)

print(resultado_pca.round(4))

# ------------------------------------------------------------------------------
# 12. MATRIZES DE CONFUSÃO
# ------------------------------------------------------------------------------

fig, axes = plt.subplots( 1,2,figsize=(12, 10))

ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred_knn_original),
    display_labels=['no', 'yes']
).plot(
    ax=axes[0],
    cmap='Blues',
    colorbar=False
)

axes[0].set_title('KNN - Original')
axes[0].grid(False)

ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred_knn_pca),
    display_labels=['no', 'yes']
).plot(
    ax=axes[1],
    cmap='Blues',
    colorbar=False
)

axes[1].set_title('KNN - PCA')
axes[1].grid(False)

plt.tight_layout()
plt.show()