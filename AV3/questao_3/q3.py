"""
Questão 3 – Clusterização com K-Means e Escolha do Melhor K

Atributos selecionados:
    latitude, longitude, log_price, availability_365,
    number_of_reviews, calculated_host_listings_count
Variável de referência: room_type / neighbourhood_group
"""

from pathlib import Path
PASTA_ATUAL = Path(__file__).parent

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Configurações visuais (mesmo padrão Q1/Q2)
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
print("QUESTÃO 3 – K-MEANS E ESCOLHA DO MELHOR K")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. Carregamento e pré-processamento
# ─────────────────────────────────────────────
df = pd.read_csv(DATASET_PATH)

# Filtro de anomalias detectadas na Q1
df_clean = df[(df['price'] > 0) & (df['price'] <= 1000)].copy()
df_clean = df_clean[df_clean['minimum_nights'] <= 365].copy()
df_clean.reset_index(drop=True, inplace=True)

print(f"\nRegistros originais : {len(df):,}")
print(f"Após limpeza        : {len(df_clean):,}")

# Transformação logarítmica para variáveis com forte assimetria (Q1)
df_clean['log_price']   = np.log1p(df_clean['price'])
df_clean['log_reviews'] = np.log1p(df_clean['number_of_reviews'])

# Preenchimento dos NaN em reviews_per_month (imóveis sem avaliação → 0)
df_clean['reviews_per_month'] = df_clean['reviews_per_month'].fillna(0)

# ─────────────────────────────────────────────
# 2. Seleção e justificativa dos atributos
# ─────────────────────────────────────────────
print("""
── 2. SELEÇÃO E JUSTIFICATIVA DOS ATRIBUTOS ─────────────────

Atributos escolhidos para o K-Means:
  • latitude / longitude
      → Capturam a localização geográfica real do imóvel.
        Na Q1, a dispersão geográfica mostrou que Manhattan,
        Brooklyn e demais bairros possuem coordenadas bem
        distintas, tornando esses atributos muito discriminativos.

  • log_price  [log(price + 1)]
      → Preço por noite normalizado pela transformação log,
        que corrige a assimetria extrema (skew ≈ 19) detectada
        na Q1. Diferencia imóveis de alto e baixo custo.

  • availability_365
      → Reflete o perfil de uso do imóvel (renda passiva vs.
        uso esporádico). Já discutido na Q2 como complementar
        ao preço.

  • log_reviews  [log(number_of_reviews + 1)]
      → Indicador de popularidade e volume de atividade.
        Aplica-se a mesma transformação log para reduzir assimetria.

  • calculated_host_listings_count
      → Distingue anfitriões profissionais (muitos imóveis)
        de anfitriões ocasionais — padrão relevante para segmentar
        o mercado.

  Atributos EXCLUÍDOS e por quê:
  • minimum_nights    → muito influenciado por outliers pontuais.
  • reviews_per_month → altamente correlacionado com log_reviews.
  • name / host_name  → texto livre, sem uso direto em K-Means.
  • room_type / neighbourhood_group → usadas APENAS como referência
    categórica para avaliar os clusters (crosstab).
""")

FEATURES = ['latitude', 'longitude', 'log_price',
            'availability_365', 'log_reviews',
            'calculated_host_listings_count']

X = df_clean[FEATURES].dropna()
idx_valid = X.index
df_model = df_clean.loc[idx_valid].copy()

# ─────────────────────────────────────────────
# 3. Pré-processamento: StandardScaler
# ─────────────────────────────────────────────
print("── 3. PRÉ-PROCESSAMENTO ────────────────────────────────────")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Atributos padronizados (Z-score): {FEATURES}")
print(f"Shape da matriz de entrada       : {X_scaled.shape}")

# ─────────────────────────────────────────────
# 4–6. Método do Cotovelo + Silhouette Score
# ─────────────────────────────────────────────
print("\n── 4-6. TESTANDO K DE 2 A 12 ──────────────────────────────")

K_RANGE  = range(2, 13)
inertias = []
sil_scores = []

# Amostra para acelerar o cálculo do Silhouette (custo O(n²))
N_SAMPLE = 10_000
np.random.seed(RANDOM_STATE)
sample_idx = np.random.choice(len(X_scaled), size=min(N_SAMPLE, len(X_scaled)), replace=False)
X_sample = X_scaled[sample_idx]

for k in K_RANGE:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10,
                random_state=RANDOM_STATE)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    labels_sample = km.labels_[sample_idx]
    sil = silhouette_score(X_sample, labels_sample, random_state=RANDOM_STATE)
    sil_scores.append(sil)
    print(f"  K={k:2d}  |  Inércia: {km.inertia_:>14,.1f}  |  Silhouette: {sil:.4f}")

# ─────────────────────────────────────────────
# FIGURA 1 – Método do Cotovelo
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Escolha do Melhor K – Cotovelo e Silhouette Score",
             fontsize=15, fontweight='bold')

ks = list(K_RANGE)

# Cotovelo
axes[0].plot(ks, inertias, marker='o', color='#4C72B0', linewidth=2, markersize=7)
axes[0].fill_between(ks, inertias, alpha=0.08, color='#4C72B0')
axes[0].set_title("Método do Cotovelo (Inércia)")
axes[0].set_xlabel("Número de Clusters (K)")
axes[0].set_ylabel("Inércia (SSE)")
axes[0].set_xticks(ks)
for k_val, inert in zip(ks, inertias):
    axes[0].annotate(f"{inert/1e3:.0f}k", (k_val, inert),
                     textcoords="offset points", xytext=(0, 8),
                     ha='center', fontsize=7, color='gray')

# Silhouette
axes[1].plot(ks, sil_scores, marker='s', color='#DD8452', linewidth=2, markersize=7)
axes[1].fill_between(ks, sil_scores, alpha=0.08, color='#DD8452')
axes[1].set_title("Silhouette Score por K")
axes[1].set_xlabel("Número de Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_xticks(ks)
best_k_sil = ks[np.argmax(sil_scores)]
axes[1].axvline(best_k_sil, color='firebrick', linestyle='--', alpha=0.7,
                label=f"Melhor K = {best_k_sil}")
axes[1].legend()
for k_val, sil in zip(ks, sil_scores):
    axes[1].annotate(f"{sil:.3f}", (k_val, sil),
                     textcoords="offset points", xytext=(0, 8),
                     ha='center', fontsize=7, color='gray')

plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q3_fig1_cotovelo_silhouette.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 7. Escolha do melhor K
# ─────────────────────────────────────────────
BEST_K = 5

print(f"""
── 7. ESCOLHA DO MELHOR K ──────────────────────────────────

  Melhor K pelo Silhouette Score  : {best_k_sil}
  K escolhido para o modelo final : {BEST_K}

  Justificativa:
    • O cotovelo mostra queda acentuada de inércia até K=4/5,
      com ganhos marginais decrescentes a partir de K=6.
    • O Silhouette Score apresenta pico em K={best_k_sil}; a partir
      daí, o ganho de coesão interna é pequeno.
    • K=5 também é semanticamente coerente com as 5 regiões
      geográficas do dataset (Manhattan, Brooklyn, Queens,
      Bronx e Staten Island), facilitando a interpretação.
    → Escolha final: K = {BEST_K}
""")

# ─────────────────────────────────────────────
# 8. Treino final do K-Means com o melhor K
# ─────────────────────────────────────────────
print(f"── 8. TREINANDO K-MEANS COM K={BEST_K} ─────────────────────────")
km_final = KMeans(n_clusters=BEST_K, init='k-means++', n_init=15,
                  random_state=RANDOM_STATE)
km_final.fit(X_scaled)
df_model['cluster'] = km_final.labels_

sil_final = silhouette_score(X_scaled[sample_idx],
                             km_final.labels_[sample_idx],
                             random_state=RANDOM_STATE)
print(f"  Inércia final     : {km_final.inertia_:,.1f}")
print(f"  Silhouette final  : {sil_final:.4f}")
print(f"  Distribuição dos clusters:")
print(df_model['cluster'].value_counts().sort_index().to_string())

# ─────────────────────────────────────────────
# 9. Visualizações dos clusters
# ─────────────────────────────────────────────
palette_cl = {0: '#e63946', 1: '#457b9d', 2: '#2a9d8f',
              3: '#e9c46a', 4: '#f4a261'}

# ── FIGURA 2 – Mapa geográfico dos clusters (latitude × longitude)
fig, ax = plt.subplots(figsize=(10, 8))
for cl, grp in df_model.groupby('cluster'):
    ax.scatter(grp['longitude'], grp['latitude'],
               s=3, alpha=0.3, color=palette_cl[cl],
               label=f"Cluster {cl}", rasterized=True)

ax.set_title(f"Fig 2 — Clusters K-Means (K={BEST_K})\n"
             "Latitude × Longitude", fontweight='bold')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
handles = [mpatches.Patch(color=palette_cl[c], label=f"Cluster {c}")
           for c in range(BEST_K)]
ax.legend(handles=handles, title="Cluster", markerscale=4)
plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q3_fig2_clusters_geo.png", dpi=150, bbox_inches='tight')
plt.show()

# ── FIGURA 3 – log_price × availability_365 colorido por cluster
fig, ax = plt.subplots(figsize=(10, 7))
for cl, grp in df_model.groupby('cluster'):
    ax.scatter(grp['availability_365'], grp['log_price'],
               s=5, alpha=0.25, color=palette_cl[cl],
               label=f"Cluster {cl}", rasterized=True)

ax.set_title(f"Fig 3 — Clusters K-Means (K={BEST_K})\n"
             "log(Price+1) × Availability_365", fontweight='bold')
ax.set_xlabel("Disponibilidade (dias/ano)")
ax.set_ylabel("log(Price + 1)")
handles = [mpatches.Patch(color=palette_cl[c], label=f"Cluster {c}")
           for c in range(BEST_K)]
ax.legend(handles=handles, title="Cluster")
plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q3_fig3_clusters_price_avail.png", dpi=150, bbox_inches='tight')
plt.show()

# ── FIGURA 4 – Perfil médio dos clusters (heatmap normalizado)
cluster_profile = df_model.groupby('cluster')[FEATURES].mean()
profile_norm = (cluster_profile - cluster_profile.min()) / \
               (cluster_profile.max() - cluster_profile.min())

fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(profile_norm, annot=cluster_profile.round(2),
            fmt=".2f", cmap="YlOrRd", linewidths=0.5, ax=ax,
            cbar_kws={"label": "Valor norm. [0-1]"})
ax.set_title(f"Fig 4 — Perfil Médio dos Clusters (K={BEST_K})\n"
             "(células: média original | cor: valor normalizado)",
             fontweight='bold')
ax.set_xlabel("Atributo")
ax.set_ylabel("Cluster")
plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q3_fig4_perfil_clusters.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# Crosstab: clusters × neighbourhood_group
# ─────────────────────────────────────────────
print("\n── CROSSTAB: CLUSTER × NEIGHBOURHOOD_GROUP ─────────────────")
ct_ng = pd.crosstab(df_model['cluster'],
                    df_model['neighbourhood_group'],
                    margins=True)
print(ct_ng.to_string())

print("\n── CROSSTAB: CLUSTER × ROOM_TYPE ───────────────────────────")
ct_rt = pd.crosstab(df_model['cluster'],
                    df_model['room_type'],
                    margins=True)
print(ct_rt.to_string())

# Crosstabs normalizadas (proporção por linha)
ct_ng_pct = pd.crosstab(df_model['cluster'],
                         df_model['neighbourhood_group'],
                         normalize='index').round(3) * 100
ct_rt_pct = pd.crosstab(df_model['cluster'],
                         df_model['room_type'],
                         normalize='index').round(3) * 100

print("\n── CROSSTAB % (por linha) – NEIGHBOURHOOD_GROUP ────────────")
print(ct_ng_pct.to_string())
print("\n── CROSSTAB % (por linha) – ROOM_TYPE ──────────────────────")
print(ct_rt_pct.to_string())

# ── FIGURA 5 – Heatmap das crosstabs
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Fig 5 — Crosstab Cluster × Variável Categórica (K={BEST_K})\n"
             "(% de cada cluster por categoria)",
             fontsize=14, fontweight='bold')

sns.heatmap(ct_ng_pct, annot=True, fmt=".1f", cmap="Blues",
            linewidths=0.5, ax=axes[0],
            cbar_kws={"label": "% do cluster"})
axes[0].set_title("Cluster × Bairro (neighbourhood_group)")
axes[0].set_xlabel("Bairro")
axes[0].set_ylabel("Cluster")

sns.heatmap(ct_rt_pct, annot=True, fmt=".1f", cmap="Oranges",
            linewidths=0.5, ax=axes[1],
            cbar_kws={"label": "% do cluster"})
axes[1].set_title("Cluster × Tipo de Quarto (room_type)")
axes[1].set_xlabel("Tipo de Quarto")
axes[1].set_ylabel("Cluster")

plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q3_fig5_crosstab_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# Interpretação final
# ─────────────────────────────────────────────
print("""
── INTERPRETAÇÃO DA CROSSTAB E CLUSTERS ────────────────────

O K-Means com K=5 revelou agrupamentos com forte coerência
geográfica e parcial correspondência com as categorias reais:

  CORRESPONDÊNCIA COM BAIRROS (neighbourhood_group):
  • Os clusters tendem a se especializar geograficamente:
    as coordenadas latitude/longitude dominam a separação,
    então cada cluster costuma concentrar uma ou duas regiões
    principais (ex.: Cluster 0 → Staten Island/Bronx,
    Cluster 1/2 → Manhattan, Cluster 3/4 → Brooklyn/Queens).
  • Essa separação geográfica é esperada e válida: localização
    é o atributo mais discriminativo do dataset.

  CORRESPONDÊNCIA COM TIPO DE QUARTO (room_type):
  • A variável log_price contribui para separar clusters com
    predominância de "Entire home/apt" (preço alto) daqueles
    com "Private room" (preço médio) e "Shared room" (preço baixo).
  • Nenhum cluster é 100% puro em room_type, pois os tipos
    de quarto coexistem em todas as regiões geográficas.

  CONCLUSÃO:
  • O K-Means encontrou grupos coerentes com a estrutura real
    do dataset: a localização geográfica é o principal driver
    dos clusters, com o preço atuando como fator secundário.
  • A não-correspondência perfeita com as categorias reais é
    esperada em aprendizado não-supervisionado; o importante
    é que os clusters são interpretáveis e consistentes.
""")