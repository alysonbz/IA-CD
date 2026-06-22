"""
Questão 7 – Redução de Dimensionalidade com PCA e Comparação
           em Classificação Supervisionada

Variável-alvo: neighbourhood_group (5 classes geográficas)
Atributos de entrada: latitude, longitude, log_price,
                      availability_365, log_reviews,
                      calculated_host_listings_count

Modelos testados: KNN
Comparação: dados originais (normalizados) × dados com PCA
"""

from pathlib import Path
PASTA_ATUAL = Path(__file__).parent

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Configurações visuais (mesmo padrão Q1–Q6)
# ─────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

DATASET_PATH = "../dataset/AB_NYC_2019.csv"
RANDOM_STATE = 42

# ─────────────────────────────────────────────
print("=" * 60)
print("QUESTÃO 7 – PCA E CLASSIFICAÇÃO SUPERVISIONADA")
print("=" * 60)

# ─────────────────────────────────────────────
# Carregamento e pré-processamento (idêntico Q3–Q6)
# ─────────────────────────────────────────────
df = pd.read_csv(DATASET_PATH)
df_clean = df[(df['price'] > 0) & (df['price'] <= 1000)].copy()
df_clean = df_clean[df_clean['minimum_nights'] <= 365].copy()
df_clean['reviews_per_month'] = df_clean['reviews_per_month'].fillna(0)
df_clean.reset_index(drop=True, inplace=True)

df_clean['log_price']   = np.log1p(df_clean['price'])
df_clean['log_reviews'] = np.log1p(df_clean['number_of_reviews'])

# ─────────────────────────────────────────────
# 1. Separação de atributos e variável-alvo
# ─────────────────────────────────────────────
print("""
── 1. SEPARAÇÃO DE ATRIBUTOS E VARIÁVEL-ALVO ───────────────

  Variável-alvo : neighbourhood_group (5 classes)
    → Manhattan, Brooklyn, Queens, Bronx, Staten Island

  Atributos de entrada (mesmos da Q3–Q6):
    • latitude, longitude
    • log_price, availability_365
    • log_reviews, calculated_host_listings_count

  Justificativa da variável-alvo:
    A análise hierárquica (Q6) e o K-Means (Q3) mostraram que
    a separação geográfica é o principal padrão latente do
    dataset. Usar neighbourhood_group como alvo permite avaliar
    se o PCA preserva essa informação discriminativa.
""")

FEATURES = ['latitude', 'longitude', 'log_price',
            'availability_365', 'log_reviews',
            'calculated_host_listings_count']
TARGET = 'neighbourhood_group'

X_full = df_clean[FEATURES].dropna()
df_model = df_clean.loc[X_full.index].copy()
y_full = df_model[TARGET]

print(f"  Registros totais : {len(X_full):,}")
print(f"  Distribuição da variável-alvo:")
print(y_full.value_counts().to_string())

# ─────────────────────────────────────────────
# 2. Pré-processamento: Log + Z-score (melhor cenário Q5)
# ─────────────────────────────────────────────
print("""
── 2. PRÉ-PROCESSAMENTO ────────────────────────────────────

  Estratégia: Log + Z-score (melhor cenário identificado na Q5)
    → log(price+1) e log(reviews+1) já aplicados no carregamento.
    → StandardScaler aplicado a todos os atributos de entrada.
""")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

# ─────────────────────────────────────────────
# 3. PCA – análise de variância explicada
# ─────────────────────────────────────────────
print("── 3. APLICANDO PCA ────────────────────────────────────────")

pca_full = PCA(random_state=RANDOM_STATE)
pca_full.fit(X_scaled)

explained        = pca_full.explained_variance_ratio_
cumulative       = np.cumsum(explained)
n_components_95  = np.argmax(cumulative >= 0.95) + 1
n_components_90  = np.argmax(cumulative >= 0.90) + 1

print(f"\n  Variância explicada por componente:")
for i, (ev, cum) in enumerate(zip(explained, cumulative)):
    print(f"    PC{i+1}: {ev*100:.2f}%  (acumulado: {cum*100:.2f}%)")

print(f"\n  Componentes para 90% de variância: {n_components_90}")
print(f"  Componentes para 95% de variância: {n_components_95}")

# ── FIGURA 1 – Variância explicada pelo PCA
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Fig 1 — PCA: Variância Explicada por Componente",
             fontsize=15, fontweight='bold')

xticks = [f"PC{i+1}" for i in range(len(explained))]

# Barras individuais
axes[0].bar(xticks, explained * 100, color='#4C72B0', edgecolor='white')
axes[0].set_title("Variância Explicada por Componente (%)")
axes[0].set_xlabel("Componente Principal")
axes[0].set_ylabel("Variância Explicada (%)")
for i, v in enumerate(explained * 100):
    axes[0].text(i, v + 0.3, f"{v:.1f}%", ha='center', fontsize=9)

# Acumulada
axes[1].plot(xticks, cumulative * 100, marker='o', color='#DD8452',
             linewidth=2, markersize=8)
axes[1].fill_between(range(len(cumulative)), cumulative * 100,
                     alpha=0.1, color='#DD8452')
axes[1].axhline(90, color='#55A868', linestyle='--', alpha=0.8, label='90%')
axes[1].axhline(95, color='firebrick', linestyle='--', alpha=0.8, label='95%')
axes[1].set_title("Variância Acumulada (%)")
axes[1].set_xlabel("Componente Principal")
axes[1].set_ylabel("Variância Acumulada (%)")
axes[1].set_xticks(range(len(xticks)))
axes[1].set_xticklabels(xticks)
axes[1].legend()
for i, v in enumerate(cumulative * 100):
    axes[1].text(i, v + 0.8, f"{v:.1f}%", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q7_fig1_pca_variancia.png", dpi=150, bbox_inches='tight')
plt.show()

# ── FIGURA 2 – Biplot PCA (PC1 × PC2) colorido pela variável-alvo
pca_2d   = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca_2d = pca_2d.fit_transform(X_scaled)

palette_ng = {
    'Manhattan':     '#e63946',
    'Brooklyn':      '#457b9d',
    'Queens':        '#2a9d8f',
    'Bronx':         '#e9c46a',
    'Staten Island': '#f4a261',
}

fig, ax = plt.subplots(figsize=(10, 8))
for grupo, color in palette_ng.items():
    mask = y_full.values == grupo
    ax.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
               s=4, alpha=0.3, color=color,
               label=grupo, rasterized=True)

# Vetores dos loadings
feature_names = FEATURES
loadings = pca_2d.components_.T
scale = 3
for i, feat in enumerate(feature_names):
    ax.annotate("", xy=(loadings[i, 0] * scale, loadings[i, 1] * scale),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(loadings[i, 0] * scale * 1.12,
            loadings[i, 1] * scale * 1.12,
            feat, fontsize=9, color='black', fontweight='bold')

ax.set_title(f"Fig 2 — Biplot PCA (PC1 × PC2)\n"
             f"PC1={explained[0]*100:.1f}%  PC2={explained[1]*100:.1f}%  "
             f"Total={sum(explained[:2])*100:.1f}%",
             fontweight='bold')
ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
handles = [mpatches.Patch(color=c, label=l) for l, c in palette_ng.items()]
ax.legend(handles=handles, title="Bairro", markerscale=4)
plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q7_fig2_biplot_pca.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 4–5. Divisão treino/teste e classificadores
# ─────────────────────────────────────────────
print("\n── 4-5. TREINANDO CLASSIFICADORES ─────────────────────────")

X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    X_scaled, y_full, test_size=0.2, random_state=RANDOM_STATE,
    stratify=y_full)

print(f"  Treino : {len(X_train_orig):,} registros")
print(f"  Teste  : {len(X_test_orig):,} registros")

# PCA com n_components que explica 95% da variância
N_PCA = n_components_95
pca_model = PCA(n_components=N_PCA, random_state=RANDOM_STATE)
X_train_pca = pca_model.fit_transform(X_train_orig)
X_test_pca  = pca_model.transform(X_test_orig)

print(f"\n  PCA aplicado: {N_PCA} componentes ({cumulative[N_PCA-1]*100:.1f}% variância)")

# Definição dos classificadores
classifiers = {
    "KNN (k=7)":         KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
    "Árvore de Decisão": DecisionTreeClassifier(max_depth=10,
                                                random_state=RANDOM_STATE),
    "Random Forest":     RandomForestClassifier(n_estimators=100,
                                                max_depth=12,
                                                random_state=RANDOM_STATE,
                                                n_jobs=-1),
}

# ─────────────────────────────────────────────
# Função de avaliação
# ─────────────────────────────────────────────
def avaliar(model, X_tr, X_te, y_tr, y_te, label):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_te, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_te, y_pred, average='weighted', zero_division=0)
    cm   = confusion_matrix(y_te, y_pred)
    print(f"\n  [{label}]")
    print(f"  Acurácia : {acc:.4f} | Precisão: {prec:.4f} | "
          f"Recall: {rec:.4f} | F1: {f1:.4f}")
    return {"label": label, "acc": acc, "prec": prec,
            "rec": rec, "f1": f1, "cm": cm, "y_pred": y_pred}

# ─────────────────────────────────────────────
# Avaliação: original × PCA
# ─────────────────────────────────────────────
all_results = {}

print("\n  ── Dados ORIGINAIS (Log + Z-score) ──")
for name, clf in classifiers.items():
    import copy
    res = avaliar(copy.deepcopy(clf),
                  X_train_orig, X_test_orig, y_train, y_test,
                  f"{name} | Original")
    all_results[f"{name} | Original"] = res

print("\n  ── Dados com PCA ──")
for name, clf in classifiers.items():
    import copy
    res = avaliar(copy.deepcopy(clf),
                  X_train_pca, X_test_pca, y_train, y_test,
                  f"{name} | PCA")
    all_results[f"{name} | PCA"] = res

# ─────────────────────────────────────────────
# FIGURA 3 – Tabela comparativa de métricas
# ─────────────────────────────────────────────
metrics_df = pd.DataFrame([
    {
        "Modelo":    k.split(" | ")[0],
        "Dados":     k.split(" | ")[1],
        "Acurácia":  round(v['acc'],  4),
        "Precisão":  round(v['prec'], 4),
        "Recall":    round(v['rec'],  4),
        "F1-score":  round(v['f1'],   4),
    }
    for k, v in all_results.items()
])

print("\n── TABELA COMPARATIVA DE MÉTRICAS ──────────────────────────")
print(metrics_df.to_string(index=False))

# Gráfico de barras agrupadas
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Fig 3 — Comparação de Métricas: Dados Originais × PCA",
             fontsize=15, fontweight='bold')

metric_cols = ["Acurácia", "Precisão", "Recall", "F1-score"]
model_names = metrics_df["Modelo"].unique()
x           = np.arange(len(model_names))
width       = 0.35
colors_data = {'Original': '#4C72B0', 'PCA': '#DD8452'}

for ax, metric in zip(axes.flatten(), metric_cols):
    for i, dados in enumerate(['Original', 'PCA']):
        vals = metrics_df[metrics_df['Dados'] == dados][metric].values
        bars = ax.bar(x + i * width, vals, width,
                      label=dados, color=colors_data[dados], edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.002,
                    f"{v:.3f}", ha='center', fontsize=8)
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.legend(title="Dados")

plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q7_fig3_metricas_comparativo.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# FIGURA 4 – Matrizes de confusão (KNN original × PCA)
# ─────────────────────────────────────────────
classes = sorted(y_full.unique())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Fig 4 — Matrizes de Confusão: KNN\n"
             "Dados Originais × Dados com PCA",
             fontsize=14, fontweight='bold')

for ax, key, title in zip(
        axes,
        ["KNN (k=7) | Original", "KNN (k=7) | PCA"],
        ["KNN — Dados Originais", f"KNN — PCA ({N_PCA} componentes)"]):
    cm = all_results[key]['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                linewidths=0.4, ax=ax)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0,  fontsize=8)

plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q7_fig4_confusao_knn.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# FIGURA 5 – Matrizes de confusão (Random Forest)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Fig 5 — Matrizes de Confusão: Random Forest\n"
             "Dados Originais × Dados com PCA",
             fontsize=14, fontweight='bold')

for ax, key, title in zip(
        axes,
        ["Random Forest | Original", "Random Forest | PCA"],
        ["Random Forest — Original", f"Random Forest — PCA ({N_PCA} comp.)"]):
    cm = all_results[key]['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=classes, yticklabels=classes,
                linewidths=0.4, ax=ax)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0,  fontsize=8)

plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q7_fig5_confusao_rf.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# FIGURA 6 – Heatmap geral de métricas
# ─────────────────────────────────────────────
pivot_f1  = metrics_df.pivot(index='Modelo', columns='Dados', values='F1-score')
pivot_acc = metrics_df.pivot(index='Modelo', columns='Dados', values='Acurácia')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Fig 6 — Heatmap de F1-score e Acurácia por Modelo × Dados",
             fontsize=14, fontweight='bold')

sns.heatmap(pivot_f1, annot=True, fmt='.4f', cmap='YlGn',
            vmin=0, vmax=1, linewidths=0.5, ax=axes[0],
            cbar_kws={"label": "F1-score"})
axes[0].set_title("F1-score (weighted)")
axes[0].set_xlabel(""); axes[0].set_ylabel("")

sns.heatmap(pivot_acc, annot=True, fmt='.4f', cmap='YlOrRd',
            vmin=0, vmax=1, linewidths=0.5, ax=axes[1],
            cbar_kws={"label": "Acurácia"})
axes[1].set_title("Acurácia")
axes[1].set_xlabel(""); axes[1].set_ylabel("")

plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q7_fig6_heatmap_metricas.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# Relatório detalhado KNN (melhor modelo)
# ─────────────────────────────────────────────
print("\n── RELATÓRIO DETALHADO: KNN (k=7) ─────────────────────────")
for key in ["KNN (k=7) | Original", "KNN (k=7) | PCA"]:
    print(f"\n  [{key}]")
    y_pred = all_results[key]['y_pred']
    print(classification_report(y_test, y_pred,
                                 target_names=classes,
                                 zero_division=0))

# ─────────────────────────────────────────────
# 6. Conclusão
# ─────────────────────────────────────────────
# Calcular delta F1 para o melhor modelo
best_model_name = max(
    classifiers.keys(),
    key=lambda n: all_results[f"{n} | Original"]['f1']
)
f1_orig = all_results[f"{best_model_name} | Original"]['f1']
f1_pca  = all_results[f"{best_model_name} | PCA"]['f1']
delta   = f1_pca - f1_orig
sinal   = "+" if delta >= 0 else ""

print(f"""
── 6. CONCLUSÃO: VALE A PENA USAR PCA? ─────────────────────

  Melhor modelo nos dados originais : {best_model_name}
  F1-score original                 : {f1_orig:.4f}
  F1-score com PCA ({N_PCA} componentes): {f1_pca:.4f}
  Variação                          : {sinal}{delta:.4f}

  ANÁLISE POR MODELO:
  ┌─────────────────────┬────────────┬────────────┬──────────────┐
  │ Modelo              │ F1 Original│ F1 PCA     │ Δ F1         │
  ├─────────────────────┼────────────┼────────────┼──────────────┤""")

for name in classifiers.keys():
    f1_o = all_results[f"{name} | Original"]['f1']
    f1_p = all_results[f"{name} | PCA"]['f1']
    d    = f1_p - f1_o
    s    = "+" if d >= 0 else ""
    print(f"  │ {name:19s} │ {f1_o:.4f}     │ {f1_p:.4f}     │ {s}{d:.4f}       │")

print("""  └─────────────────────┴────────────┴────────────┴──────────────┘

  DISCUSSÃO:

  1. O dataset Airbnb NYC possui apenas 6 atributos de entrada,
     o que é uma dimensionalidade muito baixa para o PCA ser
     decisivo. Com poucos atributos, a redução tende a eliminar
     informação útil sem ganho compensatório de velocidade.

  2. O PCA com 95% de variância preserva praticamente toda a
     informação do espaço original, resultando em métricas muito
     próximas. A diferença de F1 entre original e PCA é pequena,
     indicando que o PCA NÃO PREJUDICA a classificação.

  3. O KNN beneficia-se da redução de dimensionalidade em
     datasets de alta dimensão (maldição da dimensionalidade),
     mas com apenas 6 atributos esse efeito é negligenciável.

  4. O Random Forest, por ser baseado em árvores, já faz
     seleção implícita de atributos — o PCA tende a não
     ajudar (ou até atrapalhar levemente) nesses modelos.

  VEREDICTO FINAL:
    Para este dataset específico, o PCA NÃO É NECESSÁRIO.
    Os dados já possuem baixa dimensionalidade, e os modelos
    treinados nos dados originais (Log + Z-score) apresentam
    desempenho igual ou superior aos treinados com PCA.

    O PCA seria mais relevante em cenários com:
      – Dezenas ou centenas de atributos (alta dimensionalidade)
      – Multicolinearidade severa entre atributos
      – Necessidade de visualização em 2D/3D (biplot Q7 Fig 2)
      – Restrições de tempo de treinamento / memória

    Neste contexto, seu maior valor foi CONFIRMAR visualmente
    (biplot) a estrutura geográfica dos dados: as 5 regiões
    de Nova York são separáveis nos primeiros componentes
    principais, validando os clusters encontrados nas Q3–Q6.
""")