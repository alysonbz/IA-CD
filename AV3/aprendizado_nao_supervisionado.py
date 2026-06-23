import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time

# ====================== LEITURA E CONCATENAÇÃO DOS DADOS =================================================
# Carregando os nomes da 561 característica
features = pd.read_csv('dataset/features.txt', sep=r'\s+', header=None, names=['index', 'feature_name'])

# No arquivo há nomes de colunas repetidos.
# Tornando os nomes duplicados em únicos.
nomes_originais = features['feature_name'].values
nomes_colunas = []
print_counts = {}

for nome in nomes_originais:
    if nome in print_counts:
        print_counts[nome] += 1
        # Se o nome já existe, adiciona o sufixo _1, _2, etc.
        nomes_colunas.append(f"{nome}_{print_counts}")
    else:
        print_counts[nome] = 0
        nomes_colunas.append(nome)

# Os dados foram disponibilizados com a divisão de treino e teste já realizadas.
# Desse modo, para as primeiras questões, iremos realizar a concatenação dos dados.
# Carregando os dados de atributos (X) de treino e teste.
X_train = pd.read_csv('dataset/train/X_train.txt', sep=r'\s+', header=None, names=nomes_colunas)
X_test = pd.read_csv('dataset/test/X_test.txt', sep=r'\s+', header=None, names=nomes_colunas)

# Concatenando X verticalmente
X_completo = pd.concat([X_train, X_test], ignore_index=True)

# Carregando as variáveis de referência Y (rótulos das atividade)
y_train = pd.read_csv('dataset/train/y_train.txt', sep=r'\s+', header=None, names=['atividade'])
y_test = pd.read_csv('dataset/test/y_test.txt', sep=r'\s+', header=None, names=['atividade'])

# Concatenando Y verticalmente
Y_completo = pd.concat([y_train, y_test], ignore_index=True)

# ====================== ANÁLISE EXPLORATÓRIA =================================================
print("\n=== QUESTÃO 01: ANÁLISE EXPLORATÓRIA DOS DADOS ===")

# --- 1. Descrição Geral do Dataset ---
print("\n1. Descrição Geral:")
linhas, colunas = X_completo.shape
print(f"O dataset possui {linhas} amostras (linhas) e {colunas} atributos numéricos (colunas).")
# Contagem agregada de quantos atributos pertencem a cada tipo de dado primitivo
contagem_tipos = X_completo.dtypes.value_counts()
print(f"\nDistribuição dos tipos de dados:")
for tipo, quantidade in contagem_tipos.items():
    print(f"Atributos do tipo [{tipo}]: {quantidade}")

# --- 2. Verificação de Problemas nos Dados ---
print("\n1. Verificação de Problemas (Valores Ausentes):")
valores_nulos = X_completo.isnull().sum().sum()
print(f"Total de valores ausentes no dataset: {valores_nulos}.")

# --- 3. Estatísticas Descritivas ---
print("\n3. Estatísticas Descritivas (Amostra das 5 primeiras colunas):")
# Exibindo apenas as primeiras colunas para não poluir o terminal
print(X_completo.iloc[:, :5].describe())

# --- 4. Visualizações Exploratórias ---
# Para a visualização, foram escolhidas algumas colunas de aceleração (X, Y, Z)
colunas_foco = nomes_colunas[:3] # Seleciona as 3 primeiras colunas

print(f"\n4. Gerando visualizações para os atributos: {list(colunas_foco)}")

# Gráficos
plt.figure(figsize=(15, 5))

for i, col in enumerate(colunas_foco, 1):
    plt.subplot(1, 3, i)
    # Criando um gráfico de distribuição (Histograma + Densidade)
    sns.histplot(X_completo[col], kde=True, bins=30, color='purple')
    plt.title(f'Distribuição de:\n{col}')
    plt.xlabel('Valor Normalizado')
    plt.ylabel('Frequência')

plt.tight_layout()
plt.show()

# ====================== ANÁLISE VISUAL COM DOIS ATRIBUTOS =================================================
print("\n=== QUESTÃO 02: ANÁLISE VISUAL COM DOIS ATRIBUTOS ===")

# --- 1. Justificativa da escolha dos atributos ---
# Atributos selecionados:
# tBodyAcc-mean()-X (média de aceleração): Qual o movimento médio das atividades que o índividuo fez.
# tBodyAcc-std-X (desvio padrão da aceleração): Informa a intensidade de oscilação que a atividade informou.
# Elas conseguem resumir características importantes do movimento humano e são características complementares.

# --- 2. Criação do gráfico de dispersão ---
dados = pd.concat([X_completo, Y_completo], axis=1).copy()

atividades = {
    1: 'WALKING',
    2: 'WALKING_UPSTAIRS',
    3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING',
    5: 'STANDING',
    6: 'LAYING'
}

dados['atividade'] = dados['atividade'].map(atividades)

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=dados,
    x='tBodyAcc-mean()-X',
    y='tBodyAcc-std()-X',
    hue='atividade',
    alpha=0.7
)

plt.title('Dispersão entre tBodyAcc-mean()-X e tBodyAcc-std()-X')
plt.xlabel('tBodyAcc-mean()-X')
plt.ylabel('tBodyAcc-std()-X')

plt.legend(
    title='Atividade',
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)

plt.tight_layout()
plt.show()

# --- 3. Observar se há separação visual entre possíveis grupos ---
#  Há separação de sobreposição, indicando que grupos como SITTING e STANDING tenham valores parecidos do mesmo modo com WALKING e WALKING_UPSTAIRS.
# Mas se for dada a análise, dividida em dois grupos, de atividades dinâmicas (Walking, Walking Upstairs e Walking Downstairs), que requerem movimento
# para atividades estáticas (Sitting, Standing e Laying), que não precisam de grande movimentação.
# Pode-se notar a têndencia de grande oscilação em atividades dinâmicas enquanto ocorre o oposto nas estáticas que aproximam de -1
# que indica menores movimentos possíveis.

# --- 4. Caso exista uma variável categórica de referência, colorir os pontos por essa variável apenas para fins de análise visual ---
# Segundo a observações do gráfico, todas as atividades (variável categórica) foram dadas cores representando cada uma.

# --- 5. Discutir se os dois atributos escolhidos parecem suficientes para representar a estrutura dos dados ---
# Os dois atributos analisados fornecem uma boa visão inicial da estrutura dos dados, mas caso precise de análises
# complexas, outros critérios de análise como a rotação em diferentes eixos deverão ser precisas.

# ====================== Clusterização com K-Means e Escolha do Melhor K =================================================
print("\n=== QUESTÃO 03: Clusterização com K-Means e Escolha do Melhor K ===")
# ----- 1 e 2. Seleção e Justificativa de Atributos ------
# Para capturar o movimento tridimencional completo, foi selecionado as médias e os desvios padrão da aceleração do corpo nos três eixos (X, Y, Z).
atributos_kmeans = [
    'tBodyAcc-mean()-X', 'tBodyAcc-mean()-Y', 'tBodyAcc-mean()-Z',
    'tBodyAcc-std()-X', 'tBodyAcc-std()-Y', 'tBodyAcc-std()-Z']
X_selecionado = X_completo[atributos_kmeans]

# ----- 3. Pré-processamento ------
# Conforme identificado na Questão 1, os dados originais já estão previamente normalizados no intervalo [-1, 1].
# Portanto, nenhuma transformação adicional é estritamente necessária neste momento.

# ----- 4, 5 e 6. Testando múltiplos K, Método do cotovelo e Silhouette Score ------
inercias = []
silhuetas = []
valores_k = range(2, 8) # testando de 2 a 7 clusters

print("Calculando Inércias e Silhouette Score para diferentes valores de K.")
for k in valores_k:
    kmeans_teste = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_teste = kmeans_teste.fit_predict(X_selecionado)
    inercias.append(kmeans_teste.inertia_)

    # O cálculo do Silhouette Score é pesado; utilizamos uma amostra para manter a execução rápida.
    score_s = silhouette_score(X_selecionado, labels_teste, sample_size=2000, random_state=42)
    silhuetas.append(score_s)
    print(f"K = {k} | Inércia: {inercias[-1]:.2f} | Silhouette Score: {silhuetas[-1]:.4f}")

# Plotando os gráficos de avaliação
plt.figure(figsize=(12, 5))

# Gráfico do Cotovelo (Inércia)
plt.subplot(1, 2, 1)
plt.plot(valores_k, inercias, marker='o', color='purple', linestyle='--')
plt.title('Método do Cotovelo (Inércia)')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Silhouette Score Médio')


# Análise do Silhouette Score
plt.subplot(1, 2, 2)
plt.plot(valores_k, silhuetas, marker='o', color='red', linestyle='-')
plt.title('Análise do Silhouette Score')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Silhouette Score Médio')
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

# ----- 7 e 8. Escolha do melhor K e Treinamento Final ------
# Sabendo que o dataset possui originalmente 6 atividades (K=6) e avaliando as métricas,
# iremos treinar o modelo definitivo com K = 6 para comparar diretamente com os rótulos reais.
melhor_k = 6
print(f"\nTreinando o K-Means com K = {melhor_k}")
kmeans_final = KMeans(n_clusters=melhor_k, random_state=42, n_init=10)
dados['cluster_kmeans'] = kmeans_final.fit_predict(X_selecionado)

# ----- 9. Visualização dos clusters obtidos ------
plt.figure(figsize=(10,6))
sns.scatterplot(
    data=dados,
    x='tBodyAcc-mean()-X',
    y='tBodyAcc-std()-X',
    hue='cluster_kmeans',
    palette='Set1',
    alpha=0.7
)
plt.title(f"Clusters identificados pelo K-Means (K={melhor_k})")
plt.xlabel('tBodyAcc-mean()-X')
plt.ylabel('tBodyAcc-std()-X')
plt.legend(title='Cluster IA', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show

# ----- 10. Tabela Cruzada (Crosstab) e Interpretação ------
print("\nTABELA CRUZADA: Clusters de IA vs Atividades Reais")
tabela_cruzada = pd.crosstab(dados['atividade'], dados['cluster_kmeans'])
print(tabela_cruzada)

# ====================== Comparação entre K-Means e DBSCAN =================================================
print("\n=== QUESTÃO 04: Comparação entre K-Means e DBSCAN ===")

atributos_dbscan = [
    'tBodyAcc-mean()-X', 'tBodyAcc-mean()-Y', 'tBodyAcc-mean()-Z',
    'tBodyAcc-std()-X', 'tBodyAcc-std()-Y', 'tBodyAcc-std()-Z'
]

X_dbscan = X_completo[atributos_dbscan]

# ----- 1. Explicar brevemente a diferença entre K-Means e DBSCAN ------
# O K-Means agrupa os dados em torno de centroides e exige que o número de
# clusters seja definido previamente. Já o DBSCAN forma grupos com base na
# densidade dos pontos, não necessitando informar a quantidade de clusters e
# sendo capaz de identificar observações consideradas ruído. Enquanto o K-Means
# funciona melhor para grupos com formatos mais regulares, o DBSCAN consegue detectar
# agrupamentos de formatos variados.

# ----- 2. Escolher valores adequados para os parâmetros eps (define raio de vizinhança) e min_samples (quantidade mínima de vizinhos) ------
print("\nEscolher melhor eps:")

for eps in [0.2, 0.3, 0.4, 0.5, 0.6]:
    db = DBSCAN(eps=eps, min_samples=10)
    labels = db.fit_predict(X_dbscan)

    clusters = len(set(labels)) - (1 if -1 in labels else 0)
    ruido = np.sum(labels == -1)

    print(f"eps={eps} | clusters={clusters} | ruido={ruido}")

dbscan = DBSCAN(
    eps=0.2,
    min_samples=10
)

# ----- 3. Justificar como esses parâmetros foram definidos ------
# Como os atributos encontram-se normalizados no intervalo [-1,1], foi
# adotado inicialmente eps=0.4 para definir a vizinhança dos pontos. O parâmetro
# min_samples=10 foi escolhido para evitar a formação de clusters a partir de pequenas
# concentrações ocasionais de observações.

# ----- 4. Aplicar o DBSCAN ------
dados['cluster_dbscan'] = dbscan.fit_predict(X_dbscan)

# ----- 5. Identificar: ------
# Quantidade de clusters encontrados
print("\nCaso eps=0.20")

n_clusters = len(set(dados['cluster_dbscan'])) - (
    1 if -1 in dados['cluster_dbscan'].unique() else 0
)

print(f"Clusters encontrados: {n_clusters}")
# Quantidades de pontos considerados ruídos
ruidos = (dados['cluster_dbscan'] == -1).sum()

print(f"Quantidade de pontos classificados como ruído: {ruidos}")

# ----- 6. Visualizar os agrupamentos gerados ------
plt.figure(figsize=(10,6))

sns.scatterplot(
    data=dados,
    x='tBodyAcc-mean()-X',
    y='tBodyAcc-std()-X',
    hue='cluster_dbscan',
    palette='tab10',
    alpha=0.7
)

plt.title('Clusters encontrados pelo DBSCAN')
plt.show()

# ----- 7. Comparar os resultados com o K-Means ------
comparacao = pd.DataFrame({
    "Critério": [
        "Quantidade de clusters",
        "Quantidade de ruídos",
        "Formato dos agrupamentos",
        "Sensibilidade à escala",
        "Necessidade de definir K",
        "Facilidade de interpretação"
    ],

    "K-Means": [
        "6",
        "0",
        "Circular",
        "Alta",
        "Sim",
        "Alta"
    ],

    "DBSCAN": [
        "2",
        "204",
        "Qualquer ",
        "Alta",
        "Não",
        "Média"
    ]
})

print("\n=== COMPARAÇÃO K-MEANS x DBSCAN ===\n")
print(comparacao.to_string(index=False))


# ====================== QUESTÃO 05: IMPACTO DA NORMALIZAÇÃO =================================================
print("\n=== QUESTÃO 05: IMPACTO DA NORMALIZAÇÃO NA CLUSTERIZAÇÃO ===")

# --- 1. Aplicação da Normalização Z-score (StandardScaler) ---
# Como os dados originais já estão em formato [-1, 1], vamos aplicar o StandardScaler
# para avaliar o impacto da padronização por Z-score (Média=0, Variância=1).
scaler = StandardScaler()
X_escalado = scaler.fit_transform(X_selecionado)

# Convertemos para DataFrame para manipulação segura de colunas
X_escalado_df = pd.DataFrame(X_escalado, columns=atributos_kmeans)

# --- 2. Treinamento do K-Means com os Novos Dados Escalados ---
print("A treinar o K-Means final (K=6) com dados normalizados por Z-score.")
kmeans_zscore = KMeans(n_clusters=melhor_k, random_state=42, n_init=10)
dados['cluster_kmeans_zscore'] = kmeans_zscore.fit_predict(X_escalado_df)

# Guardamos as novas variáveis no DataFrame principal apenas para a plotagem do gráfico
dados['mean_X_zscore'] = X_escalado_df['tBodyAcc-mean()-X']
dados['std_X_zscore'] = X_escalado_df['tBodyAcc-std()-X']

# --- 3. Cálculo do Novo Silhouette Score ---
score_s_zscore = silhouette_score(X_escalado_df, dados['cluster_kmeans_zscore'], sample_size=2000, random_state=42)

# Resgatando o score do K anterior (K=6) calculado na Questão 3 para comparação
score_original = silhuetas[valores_k.index(6)]

print(f"\n--- COMPARAÇÃO DE MÉTRICAS (K=6) ---")
print(f"Silhouette Score (Normalização Original [-1, 1]): {score_original:.4f}")
print(f"Silhouette Score (Nova Normalização Z-score): {score_s_zscore:.4f}")

# --- 4. Geração do Novo Gráfico de Dispersão ---
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=dados,
    x='mean_X_zscore',
    y='std_X_zscore',
    hue='cluster_kmeans_zscore',
    palette='Set2',
    alpha=0.7
)
plt.title('Clusters Identificados pelo K-Means após Normalização Z-score')
plt.xlabel('tBodyAcc-mean()-X (Escala Z-score)')
plt.ylabel('tBodyAcc-std()-X (Escala Z-score)')
plt.legend(title='Cluster Z-score', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- AVALIAÇÃO DA TRANSFORMAÇÃO LOGARÍTMICA ---
# A aplicação transformação logarítmica foi avaliada, mas descartada para este dataset pelos seguintes motivos fundamentais:
#
# 1. Restrição Matemática: O operador logarítmico exige valores estritamente maiores que zero (x > 0).
#    Como a análise exploratória revelou que os dados já vieram pré-normalizados no intervalo [-1, 1],
#    a presença de valores negativos e nulos inviabiliza a operação (geraria NaN).
#
# 2. Natureza dos Sinais: O logaritmo é projetado para contrair dados com assimetria de cauda longa.
#    Sinais de acelerômetro e giroscópio são ondas biomecânicas naturalmente simétricas em torno de uma média.
#    Aplicar o logaritmo distorceria as distâncias euclidianas do K-Means.
#
# Conclusão: O dataset não atende à condição do enunciado ("caso existam atributos com forte assimetria
# e valores positivos"), justificando tecnicamente a não aplicação da técnica.

# --- 5. Tabela Cruzada (Crosstab) após Nova Normalização ---
print("\nTABELA CRUZADA APÓS NORMALIZAÇÃO Z-SCORE:")
tabela_cruzada_zscore = pd.crosstab(dados['atividade'], dados['cluster_kmeans_zscore'])
print(tabela_cruzada_zscore)

# ====================== QUESTÃO 06: CLUSTERIZAÇÃO HIERÁRQUICA E DENDROGRAMA =================================================
print("\n=== QUESTÃO 06: CLUSTERIZAÇÃO HIERÁRQUICA E DENDROGRAMA ===")

# --- 1 e 2. Seleção de Atributos e Normalização ---
# Utilizaremos os mesmos 6 atributos tridimensionais das questões anteriores para manter a consistência.
# Extraímos uma amostra aleatória fixa (n=150) para viabilizar a renderização legível do Dendrograma.
X_hierarquico = X_selecionado.sample(n=150, random_state=42)

# --- 3, 4 e 5. Testar Métodos de Ligação (Single, Complete, Average) e Gerar os Dendrogramas ---
metodos = ['single', 'complete', 'average']
plt.figure(figsize=(18, 6))

for i, metodo in enumerate(metodos, 1):
    plt.subplot(1, 3, i)

    # Calcula a matriz de ligação utilizando a distância Euclidiana
    Z = linkage(X_hierarquico, method=metodo, metric='euclidean')

    # Plota o dendrograma correspondente
    dendrogram(Z, no_labels=True, color_threshold=1.5)

    plt.title(f'Método de Ligação: {metodo.upper()}')
    plt.xlabel('Índice das Amostras')
    plt.ylabel('Distância Euclidiana')

plt.tight_layout()
plt.show()


# ====================== QUESTÃO 07: PCA E CLASSIFICAÇÃO SUPERVISIONADA =====================================
print("\n=== QUESTÃO 07: PCA E CLASSIFICAÇÃO SUPERVISIONADA ===")

# --- 1. Carregamento dos Rótulos (y) de Treino e Teste ---
# Como a questão envolve classificação supervisionada, precisamos dos rótulos reais
y_train = pd.read_csv('dataset/train/y_train.txt', header=None)[0]
y_test = pd.read_csv('dataset/test/y_test.txt', header=None)[0]

# --- 2. Pré-processamento Obrigatório para o PCA: Padronização ---
# O PCA é extremamente sensível à escala. Aplicamos o StandardScaler em todas as 561 colunas.
scaler_pca = StandardScaler()
X_train_scaled = scaler_pca.fit_transform(X_train)
X_test_scaled = scaler_pca.transform(X_test)

# --- 3. Treinamento com TODOS os Atributos Originais (Sem PCA) ---
print("\n[Cenário 1] Treinando classificador com todos os 561 atributos.")
knn_original = KNeighborsClassifier(n_neighbors=5)

t_ini = time.time()
knn_original.fit(X_train_scaled, y_train)
t_fim_train_orig = time.time() - t_ini

y_pred_orig = knn_original.predict(X_test_scaled)
acc_original = accuracy_score(y_test, y_pred_orig)

print(f"   - Tempo de Treinamento: {t_fim_train_orig:.4f} segundos")
print(f"   - Acurácia Global: {acc_original:.4f}")

# --- 4. Aplicação do PCA para Redução de Dimensionalidade ---
# Configura o PCA para reter 95% da variância explicada acumulada dos dados
pca = PCA(n_components=0.95, random_state=42)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

n_componentes_escolhidos = pca.n_components_
print(f"\n[PCA] Número de componentes necessários para reter 95% da variância: {n_componentes_escolhidos}")

# --- 5. Treinamento COM os Componentes do PCA ---
print(f"\n[Cenário 2] Treinando classificador com os {n_componentes_escolhidos} componentes do PCA.")
knn_pca = KNeighborsClassifier(n_neighbors=5)

t_ini = time.time()
knn_pca.fit(X_train_pca, y_train)
t_fim_train_pca = time.time() - t_ini

y_pred_pca = knn_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)

print(f"   - Tempo de Treinamento: {t_fim_train_pca:.4f} segundos")
print(f"   - Acurácia Global com PCA: {acc_pca:.4f}")

# --- 6. Relatório Detalhado de Métricas (Cenário PCA) ---
print("\n>>> Relatório de Classificação Detalhado (Com PCA):")
print(classification_report(y_test, y_pred_pca))

# --- 7. Matriz de Confusão ---
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_pca), annot=True, fmt='d', cmap='Blues')
plt.title(f'Matriz de Confusão - KNN com PCA ({n_componentes_escolhidos} componentes)')
plt.xlabel('Previsão do Modelo')
plt.ylabel('Rótulo Real')
plt.tight_layout()
plt.show()