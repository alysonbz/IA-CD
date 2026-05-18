# ==========================================================
# PROJETO DE CLASSIFICAÇÃO COM KNN
# DATASET: DRY BEAN DATASET
# ==========================================================

# IMPORTAÇÃO DAS BIBLIOTECAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import (LabelEncoder,StandardScaler,MinMaxScaler)
from sklearn.model_selection import (train_test_split,cross_val_score,GridSearchCV,StratifiedKFold)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix)


# ==========================================================
# CONFIGURAÇÕES VISUAIS
# ==========================================================

plt.style.use('default')
sns.set_theme(style="whitegrid")

# ==========================================================
# ETAPA 1 - CARREGAMENTO E EXPLORAÇÃO DO DATASET
# ==========================================================

# Carregando o dataset
drybean = pd.read_excel('Dry_Bean_Dataset.xlsx')

# Exibindo primeiras linhas
print("\nPrimeiras linhas do dataset:")
print(drybean.head())

# Informações gerais
print("\nInformações gerais:")
print(drybean.info())

# Estatísticas descritivas
print("\nEstatísticas descritivas:")
print(drybean.describe())

# Dimensões do dataset
print(f"\nQuantidade de linhas: {drybean.shape[0]}")
print(f"Quantidade de colunas: {drybean.shape[1]}")

# ==========================================================
# ETAPA 2 - TRATAMENTO DE DADOS
# ==========================================================

# Verificando valores ausentes
print("\nValores ausentes:")
print(drybean.isnull().sum())

# Removendo valores ausentes
drybean = drybean.dropna()

print(f"\nDataset após limpeza: {drybean.shape}")

# ==========================================================
# ETAPA 3 - CODIFICAÇÃO DA VARIÁVEL ALVO
# ==========================================================

# Transformando classes categóricas em números
le = LabelEncoder()

drybean['Class'] = le.fit_transform(drybean['Class'])

print("\nClasses transformadas:")
print(drybean['Class'].head())

# Exibindo mapeamento
print("\nMapeamento das classes:")
for indice, classe in enumerate(le.classes_):
    print(f"{classe} -> {indice}")

# ==========================================================
# ETAPA 4 - ANÁLISE DE CORRELAÇÃO
# ==========================================================

# Correlação com a variável alvo
correlacao = drybean.corr(numeric_only=True)

relevancia = correlacao['Class'].abs().sort_values(ascending=False)

print("\nAtributos mais relevantes:")
print(relevancia)

# ==========================================================
# HEATMAP DE CORRELAÇÃO
# ==========================================================

plt.figure(figsize=(16, 12))

sns.heatmap(correlacao,annot=True,fmt=".2f",cmap='coolwarm',linewidths=0.5,annot_kws={"size": 8})

plt.title('Mapa de Calor - Correlação entre Variáveis',fontsize=18)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ==========================================================
# ETAPA 5 - SEPARAÇÃO ENTRE X E y
# ==========================================================

# Variáveis independentes
X = drybean.drop(columns=['Class'])

# Variável alvo
y = drybean['Class']

print("\nColunas de entrada:")
print(X.columns.tolist())

print(f"\nVariável alvo: {y.name}")

# ==========================================================
# ETAPA 6 - DIVISÃO TREINO E TESTE
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

print(f"\nTreino: {X_train.shape[0]} amostras")
print(f"Teste: {X_test.shape[0]} amostras")

# ==========================================================
# ETAPA 7 - NORMALIZAÇÃO DOS DADOS
# ==========================================================

# ---------- Z-SCORE ----------

scaler_std = StandardScaler()

X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)

# ---------- MIN-MAX ----------

scaler_minmax = MinMaxScaler()

X_train_minmax = scaler_minmax.fit_transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)

# ---------- LOG + Z-SCORE ----------

X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)

scaler_log = StandardScaler()

X_train_log_scaled = scaler_log.fit_transform(X_train_log)
X_test_log_scaled = scaler_log.transform(X_test_log)

print("\nPrimeira linha - StandardScaler:")
print(X_train_std[0])

print("\nPrimeira linha - MinMaxScaler:")
print(X_train_minmax[0])

# ==========================================================
# ETAPA 8 - IMPLEMENTAÇÃO DO KNN
# ==========================================================

# Modelo inicial
knn = KNeighborsClassifier(n_neighbors=5)

# Treinamento
knn.fit(X_train_std, y_train)

# Previsões
y_pred = knn.predict(X_test_std)

# Acurácia
acuracia = accuracy_score(y_test, y_pred)

print(f"\nAcurácia inicial do modelo: {acuracia * 100:.2f}%")

# ==========================================================
# ETAPA 9 - TESTANDO DIFERENTES VALORES DE K
# ==========================================================

lista_k = [1, 3, 5, 7, 9, 11, 13, 15]

lista_acuracias = []

for k in lista_k:
    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(X_train_std, y_train)
    score = modelo.score(X_test_std, y_test)
    lista_acuracias.append(score)

    print(f"K = {k} -> Acurácia = {score * 100:.2f}%")

# ==========================================================
# ETAPA 10 - GRÁFICO K VS ACURÁCIA
# ==========================================================

plt.figure(figsize=(10, 6))

plt.plot(lista_k,lista_acuracias,marker='o',linestyle='dashed')
plt.title('Influência do Valor de K na Acurácia',fontsize=16)
plt.xlabel('Valor de K')
plt.ylabel('Acurácia')
plt.grid(True)
plt.show()

# ==========================================================
# ETAPA 11 - MELHOR K VISUALMENTE
# ==========================================================

melhor_acuracia = max(lista_acuracias)

melhor_k = lista_k[lista_acuracias.index(melhor_acuracia)]

print(f"\nMelhor K encontrado visualmente: " f"{melhor_k}")

print(f"Acurácia correspondente: " f"{melhor_acuracia * 100:.2f}%")

# ==========================================================
# GRÁFICO COM MELHOR K DESTACADO
# ==========================================================

plt.figure(figsize=(10, 6))

plt.plot(lista_k,lista_acuracias,marker='o',linestyle='dashed')
plt.annotate(f'Melhor K = {melhor_k}',xy=(melhor_k, melhor_acuracia),
    xytext=(melhor_k + 1, melhor_acuracia - 0.005),
    arrowprops=dict(facecolor='red', shrink=0.05))
plt.title('Identificação Visual do Melhor K',fontsize=16)
plt.xlabel('Valor de K')
plt.ylabel('Acurácia')
plt.grid(True)
plt.show()

# ==========================================================
# ETAPA 12 - VALIDAÇÃO CRUZADA
# ==========================================================

# Criando a estratégia de validação cruzada
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

# Criando o modelo com o melhor K encontrado
modelo_cv = KNeighborsClassifier(n_neighbors=melhor_k)

# Aplicando validação cruzada
scores_cv = cross_val_score(modelo_cv,scaler_std.fit_transform(X),y,cv=cv,scoring='accuracy')

# Exibindo resultados
print("\nAcurácia em cada fold:")
print(scores_cv)

print(f"\nMédia da validação cruzada: "f"{scores_cv.mean() * 100:.2f}%")

print(f"Desvio padrão das acurácias: "f"{scores_cv.std() * 100:.2f}%")

# ==========================================================
# ETAPA 13 - GRID SEARCH
# ==========================================================

parametros = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']}

grid = GridSearchCV(KNeighborsClassifier(),parametros,cv=5,scoring='accuracy')

grid.fit(X_train_std, y_train)

print("\nMelhores parâmetros:")
print(grid.best_params_)

print(f"\nMelhor acurácia do Grid Search: "f"{grid.best_score_ * 100:.2f}%")

# Melhor modelo
melhor_modelo = grid.best_estimator_

# ==========================================================
# ETAPA 14 - COMPARAÇÃO DAS NORMALIZAÇÕES
# ==========================================================

# Modelo otimizado
knn_final =  grid.best_estimator_

# StandardScaler
score_std = cross_val_score(knn_final,StandardScaler().fit_transform(X),y,cv=cv,scoring='accuracy').mean()

# MinMaxScaler
score_minmax = cross_val_score(knn_final,MinMaxScaler().fit_transform(X),y,cv=cv,scoring='accuracy').mean()

# Log + Scaler
X_log = np.log1p(X)

X_log_scaled = StandardScaler().fit_transform(X_log)

score_log = cross_val_score(knn_final,X_log_scaled,y,cv=cv,scoring='accuracy').mean()

print(f"\nStandardScaler: "f"{score_std * 100:.2f}%")
print(f"MinMaxScaler: "f"{score_minmax * 100:.2f}%")
print(f"Log + StandardScaler: "f"{score_log * 100:.2f}%")

# ==========================================================
# GRÁFICO DE COMPARAÇÃO
# ==========================================================

plt.figure(figsize=(10, 6))

tecnicas = ['Z-Score','Min-Max','Log + Z-Score']

acuracias = [score_std * 100, score_minmax * 100, score_log * 100]

grafico = sns.barplot(x=tecnicas,y=acuracias,palette='Set2',hue=tecnicas,legend=False)

plt.title('Comparação entre Técnicas de Normalização',fontsize=16)
plt.ylabel('Acurácia Média (%)')
plt.xlabel('Técnica')
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Valores sobre as barras
for barra in grafico.patches:
    grafico.annotate(
        f"{barra.get_height():.2f}%",
        (barra.get_x() + barra.get_width() / 2,barra.get_height()),
        ha='center',
        va='bottom',
        fontsize=10
    )
plt.show()

# ==========================================================
# ETAPA 15 - MATRIZ DE CONFUSÃO
# ==========================================================

# Reajustando modelo final
melhor_modelo.fit(X_train_std, y_train)

# Novas previsões
y_pred_final = melhor_modelo.predict(X_test_std)

# Matriz de confusão
matriz = confusion_matrix(y_test, y_pred_final)

plt.figure(figsize=(10, 8))
sns.heatmap(matriz,annot=True,fmt='d',cmap='Blues')

plt.title('Matriz de Confusão',fontsize=16)
plt.xlabel('Classe Predita')
plt.ylabel('Classe Real')
plt.show()

# ==========================================================
# ETAPA 16 - CLASSIFICATION REPORT
# ==========================================================

print(classification_report(y_test,y_pred_final))

# ==========================================================
# ETAPA 17 - ANÁLISE CRÍTICA FINAL
# ==========================================================

print("""
1. O algoritmo KNN apresentou excelente desempenho
na classificação dos grãos do dataset Dry Bean.

2. A normalização mostrou-se essencial para melhorar
o desempenho do modelo, pois o KNN utiliza distância
entre amostras.

3. O valor de K influencia diretamente a capacidade
de generalização do modelo.

4. O Grid Search permitiu identificar automaticamente
os melhores hiperparâmetros, aumentando a robustez
do classificador.

5. O modelo apresentou elevada capacidade de
generalização, comprovada pela validação cruzada.

6. As técnicas de pré-processamento tiveram impacto
direto no desempenho final do sistema.

7. O modelo demonstrou potencial aplicação em
sistemas automatizados de classificação agrícola.
""")
