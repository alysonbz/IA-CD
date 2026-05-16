from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, FunctionTransformer, OneHotEncoder, PowerTransformer,
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("knn")

# Precisa instalar ;0
from ucimlrepo import fetch_ucirepo


LABELS = {
    'age': 'Idade',
    'sex': 'Sexo',
    'cp': 'Tipo de Dor no Peito',
    'trestbps': 'Pressão Arterial em Repouso',
    'chol': 'Colesterol Sérico',
    'fbs': 'Glicemia > 120 mg/dl',
    'restecg': 'ECG em Repouso',
    'thalach': 'Freq. Cardíaca Máxima',
    'exang': 'Angina por Exercício',
    'oldpeak': 'Depressão ST',
    'slope': 'Inclinação ST',
    'ca': 'Vasos Principais Coloridos',
    'thal': 'Talassemia',
    'target': 'Diagnóstico',
}


# -------------------------------------------------------
# 1. Carregando o dataset
# -------------------------------------------------------

def carregar_dados():
    """
    Busca o dataset Heart Disease (id=45) do UCI ML Repository via ucimlrepo.
    Junta features e target em um único DataFrame e retorna ele.
    O target original tem valores de 0 a 4 representando severidade da doença.
    """
    heart_disease = fetch_ucirepo(id=45)
    df = heart_disease.data.features.copy()
    df['target'] = heart_disease.data.targets.values
    return df


# -------------------------------------------------------
# 2. Exploração inicial
# -------------------------------------------------------

def explorar_dados(df):
    """
    Faz uma exploração inicial completa do dataset: head, describe e info.
    Imprime estatísticas descritivas das colunas numéricas e
    destaca observações relevantes como outliers e desbalanceamento de target.
    """
    print("--- Exploração Inicial ---")
    print(f"\nShape: {df.shape}")
    print("\nPrimeiras linhas:")
    print(df.head())
    print("\nEstatísticas descritivas:")
    print(df.describe().T.round(3))
    print("\nInfo:")
    df.info()


# -------------------------------------------------------
# 3. Pré-processamento
# -------------------------------------------------------

def verificar_missing(df):
    """
    Verifica e imprime a quantidade de valores ausentes por coluna.
    ca e thal possuem poucos missings - a imputação é feita dentro do pipeline
    de modelagem usando a moda do treino, evitando data leakage.
    """
    missing = df.isna().sum()
    print("\n--- Valores Ausentes ---")
    print(missing)
    return missing


def plotar_outliers(df):
    """
    Plota box plots das variáveis contínuas para inspeção visual de outliers.
    Outliers não são removidos pois os dados são confiáveis - a normalização
    é usada para mitigar o impacto deles no modelo.
    """
    cont_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for i, col in enumerate(cont_features):
        sns.boxplot(data=df, y=col, ax=axes[i], color='steelblue')
        axes[i].set_title(LABELS.get(col, col))
        axes[i].set_ylabel('Valores')
    fig.suptitle('Box plot - Variáveis Contínuas', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'outliers_boxplot.png', dpi=150)
    plt.close()
    print("salvo: outliers_boxplot.png")


def binarizar_target(df):
    """
    Converte o target de múltiplas classes (0-4) para binário (0 ou 1).
    O dataset original tem alto desbalanceamento entre as classes 1-4,
    e o KNN tende a favorecer a classe majoritária (0). A binarização
    aproxima a distribuição: 164 saudáveis vs 139 doentes.
    """
    df = df.copy()
    df['target'] = (df['target'] > 0).astype(int)
    print("\n--- Target binarizado ---")
    print(df['target'].value_counts())
    return df


# -------------------------------------------------------
# 4. Distribuição dos dados
# -------------------------------------------------------

def plotar_distribuicao_target(df):
    """
    Plota a distribuição do target original (0-4) em barras horizontais.
    Chamada ANTES da binarização para mostrar o desbalanceamento entre as 5 classes,
    que motiva a decisão de binarizar antes de treinar o KNN.
    """
    target_counts = df['target'].value_counts().sort_index()
    labels = [
        f'{i} - {"Sem doença" if i == 0 else f"Grau {i}"}' for i in target_counts.index]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        x=target_counts.values,
        y=labels,
        hue=labels,
        palette='Blues_r',
        orient='h',
        legend=False,
        ax=ax,
    )
    for bar, val in zip(ax.patches, target_counts.values):
        ax.text(
            bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
            str(val), va='center', fontsize=10,
        )
    ax.set_title('Distribuição do Target', fontsize=13)
    ax.set_xlabel('Contagem')
    ax.set_ylabel('Diagnóstico')
    sns.despine()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'distribuicao_target.png', dpi=150)
    plt.close()
    print("salvo: distribuicao_target.png")


def plotar_variaveis_categoricas(df):
    """
    Plota countplots das variáveis categóricas separados por target.
    Útil para identificar quais categorias estão mais associadas
    a diagnósticos positivos ou negativos.
    """
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        sns.countplot(data=df, x=col, hue='target', ax=axes[i], palette='Set2')
        axes[i].set_title(LABELS.get(col, col))
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Contagem')
        axes[i].legend(title='target', fontsize=7,
                       title_fontsize=7, loc='upper right')
    fig.suptitle(
        'Distribuição das Variáveis Categóricas por Target', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'distribuicao_categoricas.png', dpi=150)
    plt.close()
    print("salvo: distribuicao_categoricas.png")


def plotar_variaveis_continuas(df):
    """
    Plota histogramas com KDE das variáveis contínuas.
    Mostra a forma da distribuição de cada feature numérica.
    """
    cont_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for i, col in enumerate(cont_cols):
        sns.histplot(data=df, x=col, kde=True,
                     ax=axes[i], color='steelblue', alpha=0.7)
        axes[i].set_title(LABELS.get(col, col))
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Contagem')
    fig.suptitle('Distribuição das Variáveis Contínuas', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'distribuicao_continuas.png', dpi=150)
    plt.close()
    print("salvo: distribuicao_continuas.png")


# -------------------------------------------------------
# 5. Seleção de variáveis
# -------------------------------------------------------

def plotar_correlacao(df):
    """
    Plota heatmap de correlação de Pearson entre todas as variáveis.
    A máscara triangular superior evita duplicação. As features selecionadas
    foram as com maior correlação absoluta com target no heatmap.
    """
    fig, ax = plt.subplots(figsize=(12, 9))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        ax=ax,
        linewidths=0.5,
        square=True,
    )
    ax.set_title('Correlação entre Todas as Variáveis', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'heatmap_correlacao.png', dpi=150)
    plt.close()
    print("salvo: heatmap_correlacao.png")


def selecionar_features(df):
    """
    Seleciona as features com maior correlação com target conforme o heatmap.
    Retorna X (features selecionadas) e y (target binarizado).
    Features escolhidas: ca, thal, oldpeak, thalach, cp, exang, slope.
    """
    selected_features = [
        'ca',       # Vasos Principais Coloridos
        'thal',     # Talassemia
        'oldpeak',  # Depressão ST
        'thalach',  # Freq. Cardíaca Máxima
        'cp',       # Tipo de Dor no Peito
        'exang',    # Angina por Exercício
        'slope',    # Inclinação ST
    ]
    X = df[selected_features]
    y = df['target']
    print(f"\nFeatures selecionadas: {selected_features}")
    return X, y


# -------------------------------------------------------
# 6. Transformações nas variáveis contínuas
# -------------------------------------------------------

def plotar_continuas_transformadas(df, X_train, X_test):
    """
    Aplica transformações nas variáveis contínuas e plota as distribuições resultantes.
    log10 para trestbps e chol; log10(x+1) para oldpeak (tem zeros); PowerTransformer
    para thalach. Fit exclusivamente no treino para evitar data leakage.
    """
    cont_cols_plot = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    df_train_t = df.loc[X_train.index, cont_cols_plot].copy()
    df_test_t  = df.loc[X_test.index,  cont_cols_plot].copy()

    log10_tr = FunctionTransformer(func=np.log10, validate=False)
    df_train_t['trestbps'] = log10_tr.fit_transform(df_train_t[['trestbps']])
    df_test_t['trestbps']  = log10_tr.transform(df_test_t[['trestbps']])

    log10_ch = FunctionTransformer(func=np.log10, validate=False)
    df_train_t['chol'] = log10_ch.fit_transform(df_train_t[['chol']])
    df_test_t['chol']  = log10_ch.transform(df_test_t[['chol']])

    log10_shift = FunctionTransformer(func=lambda x: np.log10(x + 1), validate=False)
    df_train_t['oldpeak'] = log10_shift.fit_transform(df_train_t[['oldpeak']])
    df_test_t['oldpeak']  = log10_shift.transform(df_test_t[['oldpeak']])

    pt = PowerTransformer()
    df_train_t['thalach'] = pt.fit_transform(df_train_t[['thalach']])
    df_test_t['thalach']  = pt.transform(df_test_t[['thalach']])

    df_transformed = pd.concat([df_train_t, df_test_t]).sort_index()

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for i, col in enumerate(cont_cols_plot):
        sns.histplot(data=df_transformed, x=col, kde=True,
                     ax=axes[i], color='steelblue', alpha=0.7)
        axes[i].set_title(LABELS.get(col, col))
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Contagem')
    fig.suptitle('Distribuição das Variáveis Contínuas (após transformações)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'distribuicao_continuas_transformadas.png', dpi=150)
    plt.close()
    print("salvo: distribuicao_continuas_transformadas.png")


# -------------------------------------------------------
# 7. Preparação para modelagem
# -------------------------------------------------------

def build_ct(norm):
    """
    Constrói um ColumnTransformer que aplica:
    - Imputação por moda + normalização nas colunas numéricas contínuas e ordinais
    - Passthrough em features binárias (exang)
    - Imputação por moda + OneHotEncoding nas colunas nominais (cp, thal)
    A imputação usa a moda do treino, evitando data leakage.
    """
    continuous_cols = ['oldpeak', 'thalach']
    ordinal_cols = ['ca', 'slope']
    passthrough_cols = ['exang']
    nominal_cols = ['cp', 'thal']
    numeric_all = continuous_cols + ordinal_cols

    return ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('scaler', norm),
        ]), numeric_all),
        ('passthrough', 'passthrough', passthrough_cols),
        ('ohe', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
        ]), nominal_cols),
    ])


def dividir_e_escalonar(X, y):
    """
    Divide os dados em 80% treino e 20% teste com estratificação pelo target,
    garantindo proporção de classes igual em ambos os conjuntos.
    Depois aplica as três normalizações (MinMax, Standard, Log) no treino e teste,
    ajustando apenas no treino para evitar data leakage.
    Retorna X_train, X_test, y_train, y_test e o dicionário X_scaled com os dados transformados.
    """
    scalers = {
        'MinMaxScaler':   MinMaxScaler(),
        'StandardScaler': StandardScaler(),
        'Log':            FunctionTransformer(np.log1p),
    }
    k_range = list(range(1, 30))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(
        f"\nTreino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

    X_scaled = {}
    for name, norm in scalers.items():
        ct = build_ct(clone(norm))
        X_scaled[name] = {
            'train': ct.fit_transform(X_train),
            'test': ct.transform(X_test),
        }

    return X_train, X_test, y_train, y_test, X_scaled, scalers, k_range


# -------------------------------------------------------
# 8. Treinamento - split simples
# -------------------------------------------------------

def treinar_split(X_scaled, y_train, y_test, k_range):
    """
    Treina KNN para cada valor de k (1-29) e para cada normalização,
    avaliando a acurácia no conjunto de teste (split simples).
    Plota curva de acurácia por k e destaca o melhor ponto de cada normalização.
    Retorna dicionário com as acurácias por normalização.
    """
    results = {}
    for name, data in X_scaled.items():
        accs = []
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(data['train'], y_train)
            accs.append(accuracy_score(y_test, knn.predict(data['test'])))
        results[name] = accs

    fig, ax = plt.subplots(figsize=(11, 5))
    for name, accs in results.items():
        ax.plot(k_range, accs, marker='o', markersize=4, label=name)
        best_idx = int(np.argmax(accs))
        ax.scatter(k_range[best_idx], accs[best_idx], s=120, zorder=5)
        ax.annotate(
            f"k={k_range[best_idx]} ({accs[best_idx]:.2%})",
            xy=(k_range[best_idx], accs[best_idx]),
            xytext=(5, 6), textcoords='offset points', fontsize=8,
        )
    ax.set_xlabel('k')
    ax.set_ylabel('Acurácia')
    ax.set_title('Treino padrão - Acurácia por K com diferentes normalizações')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'treino_split.png', dpi=150)
    plt.close()
    print("[gráfico salvo: treino_split.png]")

    return results


# -------------------------------------------------------
# 9. Validação cruzada
# -------------------------------------------------------

def treinar_cv(X_train, y_train, scalers, k_range):
    """
    Aplica validação cruzada estratificada (5-fold) para cada valor de k
    e para cada normalização usando um Pipeline completo (prep + KNN).
    O pipeline garante que a transformação é ajustada apenas nos folds de treino.
    Resultados mais baixos que o split simples são esperados - são mais confiáveis.
    Retorna dicionário com médias de acurácia por normalização.
    """
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=18)
    cv_results = {}

    for name, norm in scalers.items():
        accs = []
        for k in k_range:
            pipe = Pipeline([
                ('prep', build_ct(clone(norm))),
                ('knn', KNeighborsClassifier(n_neighbors=k)),
            ])
            accs.append(cross_val_score(pipe, X_train, y_train,
                        cv=kfold, scoring='accuracy').mean())
        cv_results[name] = accs

    fig, ax = plt.subplots(figsize=(11, 5))
    for name, accs in cv_results.items():
        ax.plot(k_range, accs, marker='o', markersize=4, label=name)
        best_idx = int(np.argmax(accs))
        ax.scatter(k_range[best_idx], accs[best_idx], s=120, zorder=5)
        ax.annotate(
            f"k={k_range[best_idx]} ({accs[best_idx]:.2%})",
            xy=(k_range[best_idx], accs[best_idx]),
            xytext=(5, 6), textcoords='offset points', fontsize=8,
        )
    ax.set_xlabel('k')
    ax.set_ylabel('Acurácia (CV 5-fold)')
    ax.set_title(
        'Validação cruzada - Acurácia por K com diferentes normalizações')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'treino_cv.png', dpi=150)
    plt.close()
    print("salvo: treino_cv.png")

    return cv_results, kfold


# -------------------------------------------------------
# 10. Grid Search
# -------------------------------------------------------

def grid_search(X_train, X_test, y_train, y_test, scalers, kfold):
    """
    Busca automática dos melhores hiperparâmetros (k, métrica e pesos) do KNN
    usando GridSearchCV com 5-fold estratificado para cada normalização.
    O melhor modelo é selecionado pela acurácia na validação cruzada (não no teste),
    pois isso indica melhor capacidade de generalização.
    Retorna dicionário com os resultados de cada normalização.
    """
    param_grid = {
        'knn__n_neighbors': list(range(1, 30)),
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan'],
    }

    gs_results = {}
    for name, norm in scalers.items():
        pipe = Pipeline([('prep', build_ct(clone(norm))),
                        ('knn', KNeighborsClassifier())])
        gs = GridSearchCV(pipe, param_grid, cv=kfold,
                          scoring='accuracy', n_jobs=-1)
        gs.fit(X_train, y_train)
        gs_results[name] = {
            'melhor_k': gs.best_params_['knn__n_neighbors'],
            'metric':   gs.best_params_['knn__metric'],
            'weights':  gs.best_params_['knn__weights'],
            'cv_acc':   round(gs.best_score_, 4),
            'test_acc': round(gs.score(X_test, y_test), 4),
        }

    print("\n--- Grid Search Results ---")
    print(pd.DataFrame(gs_results).T.to_string())

    return gs_results


# -------------------------------------------------------
# 11. Melhor modelo - avaliação final
# -------------------------------------------------------

def avaliar_melhor_modelo(X_train, X_test, y_train, y_test, scalers, gs_results):
    """
    Seleciona a normalização com maior acurácia na validação cruzada do Grid Search,
    retreina o pipeline completo com os melhores hiperparâmetros e avalia no teste.
    Prioriza CV sobre acurácia de teste pois CV mede melhor a generalização.
    Imprime classification report e plota matriz de confusão.
    Retorna o pipeline treinado e as previsões.
    """
    best_norm_name = max(gs_results, key=lambda n: gs_results[n]['cv_acc'])
    best_gs = gs_results[best_norm_name]

    pipe_best = Pipeline([
        ('prep', build_ct(clone(scalers[best_norm_name]))),
        ('knn', KNeighborsClassifier(
            n_neighbors=best_gs['melhor_k'],
            metric=best_gs['metric'],
            weights=best_gs['weights'],
        )),
    ])
    pipe_best.fit(X_train, y_train)
    y_pred = pipe_best.predict(X_test)

    print(f"\nConfiguração selecionada: {best_norm_name} | k={best_gs['melhor_k']} "
          f"| metric={best_gs['metric']} | weights={best_gs['weights']}")
    print(classification_report(y_test, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax, colorbar=False)
    ax.set_title(
        f"Matriz de Confusão - {best_norm_name} "
        f"(k={best_gs['melhor_k']}, {best_gs['metric']})"
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'matriz_confusao.png', dpi=150)
    plt.close()
    print("salvo: matriz_confusao.png")

    return pipe_best, y_pred


# -------------------------------------------------------
# 12. Comparação: Split vs CV vs Grid Search
# -------------------------------------------------------

def comparar_abordagens(scalers, k_range, X_train, X_test, y_train, y_test,
                        cv_results, gs_results):
    """
    Monta tabela comparando as três abordagens de avaliação: split simples,
    validação cruzada e Grid Search, para cada normalização.
    O Grid Search obteve os melhores resultados por explorar mais combinações
    de hiperparâmetros. O split simples tende a superestimar a performance
    por depender de uma única divisão dos dados.
    """
    split_results = {}
    for name, norm in scalers.items():
        accs = []
        for k in k_range:
            ct = build_ct(clone(norm))
            X_tr = ct.fit_transform(X_train)
            X_te = ct.transform(X_test)
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_tr, y_train)
            accs.append(accuracy_score(y_test, knn.predict(X_te)))
        best_idx = int(np.argmax(accs))
        split_results[name] = {
            'melhor_k': k_range[best_idx], 'acc': round(accs[best_idx], 4)}

    rows = []
    for name in scalers:
        sp = split_results[name]
        cv_b = {
            'melhor_k': k_range[int(np.argmax(cv_results[name]))],
            'acc': round(max(cv_results[name]), 4),
        }
        gs = gs_results[name]
        rows.append({
            'normalização': name,
            'split_k': sp['melhor_k'],
            'split_acc': sp['acc'],
            'cv_k': cv_b['melhor_k'],
            'cv_acc': cv_b['acc'],
            'gs_k': gs['melhor_k'],
            'gs_metric': gs['metric'],
            'gs_weights': gs['weights'],
            'gs_cv_acc': gs['cv_acc'],
            'gs_test_acc': gs['test_acc'],
        })

    df_comp = pd.DataFrame(rows).set_index('normalização')
    print("\n--- Comparação: Split vs CV vs Grid Search ---")
    print(df_comp.to_string())
    return df_comp


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    """Executa o pipeline completo de classificação com KNN."""

    OUTPUT_DIR.mkdir(exist_ok=True)

    df = carregar_dados()
    explorar_dados(df)
    verificar_missing(df)
    plotar_outliers(df)

    plotar_distribuicao_target(df)
    df = binarizar_target(df)
    plotar_variaveis_categoricas(df)
    plotar_variaveis_continuas(df)

    plotar_correlacao(df)
    X, y = selecionar_features(df)

    X_train, X_test, y_train, y_test, X_scaled, scalers, k_range = dividir_e_escalonar(
        X, y)

    plotar_continuas_transformadas(df, X_train, X_test)

    split_results = treinar_split(X_scaled, y_train, y_test, k_range)
    cv_results, kfold = treinar_cv(X_train, y_train, scalers, k_range)
    gs_results = grid_search(X_train, X_test, y_train, y_test, scalers, kfold)

    avaliar_melhor_modelo(X_train, X_test, y_train,
                          y_test, scalers, gs_results)
    comparar_abordagens(scalers, k_range, X_train, X_test,
                        y_train, y_test, cv_results, gs_results)

    print("\nConcluído")


if __name__ == "__main__":
    main()
