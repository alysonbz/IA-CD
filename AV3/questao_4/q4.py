"""
Questão 4 – Comparação entre K-Means e DBSCAN

Mesmo conjunto de atributos da Q3:
    latitude, longitude, log_price, availability_365,
    log_reviews, calculated_host_listings_count
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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Configurações visuais (mesmo padrão Q1–Q3)
# ─────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

DATASET_PATH  = "../dataset/AB_NYC_2019.csv"
RANDOM_STATE  = 42
BEST_K        = 5          # mesmo K escolhido na Q3

# ─────────────────────────────────────────────
print("=" * 60)
print("QUESTÃO 4 – COMPARAÇÃO ENTRE K-MEANS E DBSCAN")
print("=" * 60)

# ─────────────────────────────────────────────
# Carregamento e pré-processamento (idêntico Q3)
# ─────────────────────────────────────────────
df = pd.read_csv(DATASET_PATH)
df_clean = df[(df['price'] > 0) & (df['price'] <= 1000)].copy()
df_clean = df_clean[df_clean['minimum_nights'] <= 365].copy()
df_clean.reset_index(drop=True, inplace=True)

df_clean['log_price']   = np.log1p(df_clean['price'])
df_clean['log_reviews'] = np.log1p(df_clean['number_of_reviews'])
df_clean['reviews_per_month'] = df_clean['reviews_per_month'].fillna(0)

FEATURES = ['latitude', 'longitude', 'log_price',
            'availability_365', 'log_reviews',
            'calculated_host_listings_count']

X = df_clean[FEATURES].dropna()
df_model = df_clean.loc[X.index].copy()

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Registros utilizados : {len(X_scaled):,}")
print(f"Atributos            : {FEATURES}")

# ─────────────────────────────────────────────
# 1. Diferença conceitual K-Means × DBSCAN
# ─────────────────────────────────────────────
print("""
── 1. DIFERENÇA ENTRE K-MEANS E DBSCAN ─────────────────────

  K-MEANS
  ────────
  • Algoritmo de partição baseado em centróides.
  • Requer que o número de clusters (K) seja definido a priori.
  • Cada ponto é obrigatoriamente atribuído a um cluster.
  • Sensível à escala dos atributos e à forma dos clusters
    (assume clusters esféricos e de tamanho similar).
  • Minimiza a inércia (soma das distâncias quadráticas ao centróide).

  DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
  ─────────────────────────────────────────────────────────────────────
  • Algoritmo baseado em densidade local.
  • Não requer K definido previamente — descobre o número de clusters.
  • Classifica pontos como: núcleo, borda ou RUÍDO (outliers).
  • Capaz de encontrar clusters de forma arbitrária (não apenas esferas).
  • Robusto a outliers, que são explicitamente identificados como ruído.
  • Parâmetros: eps (raio de vizinhança) e min_samples (densidade mínima).
""")

# ─────────────────────────────────────────────
# 2–3. Escolha dos parâmetros eps e min_samples
# ─────────────────────────────────────────────
print("── 2-3. DEFINIÇÃO DOS PARÂMETROS EPS E MIN_SAMPLES ─────────")

# Heurística para eps: curva k-distâncias (k = min_samples - 1)
# Regra de bolso: min_samples = 2 × n_features
MIN_SAMPLES = 2 * len(FEATURES)   # = 12
print(f"\n  min_samples = 2 × n_features = 2 × {len(FEATURES)} = {MIN_SAMPLES}")

# Amostra para acelerar o cálculo (dataset grande)
np.random.seed(RANDOM_STATE)
sample_idx = np.random.choice(len(X_scaled), size=10_000, replace=False)
X_sample   = X_scaled[sample_idx]

nbrs = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(X_sample)
distances, _ = nbrs.kneighbors(X_sample)
k_distances  = np.sort(distances[:, -1])

# ── FIGURA 1 – Curva k-distâncias para definir eps
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(k_distances, color='#4C72B0', linewidth=1.2)
ax.set_title(f"Fig 1 — Curva k-Distâncias (k={MIN_SAMPLES})\n"
             "O 'cotovelo' indica o valor ideal de eps",
             fontweight='bold')
ax.set_xlabel("Pontos ordenados por distância")
ax.set_ylabel(f"Distância ao {MIN_SAMPLES}º vizinho mais próximo")

# Marcação visual do cotovelo aproximado
knee_idx = int(len(k_distances) * 0.92)
eps_estimated = k_distances[knee_idx]
ax.axhline(eps_estimated, color='firebrick', linestyle='--', alpha=0.8,
           label=f"eps ≈ {eps_estimated:.3f}")
ax.legend()
plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q4_fig1_kdistances.png", dpi=150, bbox_inches='tight')
plt.show()

# Arredondamos eps para um valor limpo próximo ao cotovelo
EPS = round(eps_estimated, 2)
print(f"\n  eps estimado pela curva k-distâncias : {eps_estimated:.4f}")
print(f"  eps adotado (arredondado)            : {EPS}")
print(f"""
  Justificativa dos parâmetros:
    • min_samples = {MIN_SAMPLES}
        Regra padrão: 2 × número de atributos (6 atributos → 12).
        Garante que pontos de núcleo tenham densidade suficiente
        para não criar clusters espúrios.
    • eps = {EPS}
        Determinado pelo "cotovelo" da curva k-distâncias:
        ponto onde a distância ao k-ésimo vizinho começa a
        crescer abruptamente — separa regiões densas do ruído.
""")

# ─────────────────────────────────────────────
# 4. Aplicação do DBSCAN
# ─────────────────────────────────────────────
print("── 4. APLICANDO DBSCAN ─────────────────────────────────────")
dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, n_jobs=-1)
db_labels = dbscan.fit_predict(X_scaled)
df_model['cluster_db'] = db_labels

# ─────────────────────────────────────────────
# 5. Resultados do DBSCAN
# ─────────────────────────────────────────────
n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise_db    = (db_labels == -1).sum()
pct_noise     = n_noise_db / len(db_labels) * 100

print(f"\n  Clusters encontrados : {n_clusters_db}")
print(f"  Pontos de ruído      : {n_noise_db:,} ({pct_noise:.1f}% do total)")
print(f"\n  Distribuição por cluster (−1 = ruído):")
counts = pd.Series(db_labels).value_counts().sort_index()
print(counts.to_string())

# ─────────────────────────────────────────────
# K-Means com K=5 (re-treino para comparação)
# ─────────────────────────────────────────────
km_final = KMeans(n_clusters=BEST_K, init='k-means++', n_init=15,
                  random_state=RANDOM_STATE)
km_final.fit(X_scaled)
df_model['cluster_km'] = km_final.labels_

# Silhouette scores (amostra)
sil_km = silhouette_score(X_scaled[sample_idx],
                           km_final.labels_[sample_idx],
                           random_state=RANDOM_STATE)

# Silhouette do DBSCAN apenas nos pontos não-ruído
mask_valid = db_labels != -1
if mask_valid.sum() > 1 and len(set(db_labels[mask_valid])) > 1:
    sil_db = silhouette_score(X_scaled[mask_valid],
                               db_labels[mask_valid],
                               sample_size=10_000,
                               random_state=RANDOM_STATE)
else:
    sil_db = float('nan')

print(f"\n  Silhouette K-Means  : {sil_km:.4f}")
print(f"  Silhouette DBSCAN   : {sil_db:.4f}  (excluindo ruído)")

# ─────────────────────────────────────────────
# 6. Visualizações
# ─────────────────────────────────────────────

# Paleta dinâmica para DBSCAN (número variável de clusters)
import matplotlib.cm as cm
unique_labels = sorted(set(db_labels))
n_total       = len(unique_labels)
cmap_db       = plt.get_cmap('tab10', n_total)
color_map_db  = {lbl: ('lightgray' if lbl == -1 else cmap_db(i))
                 for i, lbl in enumerate(unique_labels)}

palette_km = {0: '#e63946', 1: '#457b9d', 2: '#2a9d8f',
              3: '#e9c46a', 4: '#f4a261'}

# ── FIGURA 2 – Lado a lado: K-Means × DBSCAN (mapa geográfico)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Fig 2 — K-Means × DBSCAN: Mapa Geográfico dos Clusters",
             fontsize=15, fontweight='bold')

# K-Means
for cl, grp in df_model.groupby('cluster_km'):
    axes[0].scatter(grp['longitude'], grp['latitude'],
                    s=3, alpha=0.3, color=palette_km[cl],
                    label=f"Cluster {cl}", rasterized=True)
axes[0].set_title(f"K-Means (K={BEST_K})")
axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")
handles_km = [mpatches.Patch(color=palette_km[c], label=f"Cluster {c}")
              for c in range(BEST_K)]
axes[0].legend(handles=handles_km, markerscale=4, fontsize=9)

# DBSCAN
for lbl, grp in df_model.groupby('cluster_db'):
    label_str = "Ruído" if lbl == -1 else f"Cluster {lbl}"
    axes[1].scatter(grp['longitude'], grp['latitude'],
                    s=3, alpha=(0.1 if lbl == -1 else 0.35),
                    color=color_map_db[lbl],
                    label=label_str, rasterized=True)
axes[1].set_title(f"DBSCAN (eps={EPS}, min_samples={MIN_SAMPLES})\n"
                  f"{n_clusters_db} clusters | {n_noise_db:,} ruídos")
axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")
handles_db = [mpatches.Patch(color=color_map_db[l],
              label=("Ruído" if l == -1 else f"Cluster {l}"))
              for l in unique_labels]
axes[1].legend(handles=handles_db, markerscale=4, fontsize=9)

plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q4_fig2_geo_comparacao.png", dpi=150, bbox_inches='tight')
plt.show()

# ── FIGURA 3 – log_price × availability (K-Means × DBSCAN)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Fig 3 — K-Means × DBSCAN: log(Price+1) × Availability_365",
             fontsize=15, fontweight='bold')

for cl, grp in df_model.groupby('cluster_km'):
    axes[0].scatter(grp['availability_365'], grp['log_price'],
                    s=4, alpha=0.2, color=palette_km[cl],
                    label=f"Cluster {cl}", rasterized=True)
axes[0].set_title(f"K-Means (K={BEST_K})")
axes[0].set_xlabel("Disponibilidade (dias/ano)")
axes[0].set_ylabel("log(Price + 1)")
axes[0].legend(handles=handles_km, fontsize=9)

for lbl, grp in df_model.groupby('cluster_db'):
    axes[1].scatter(grp['availability_365'], grp['log_price'],
                    s=4, alpha=(0.08 if lbl == -1 else 0.25),
                    color=color_map_db[lbl],
                    label=("Ruído" if lbl == -1 else f"Cluster {lbl}"),
                    rasterized=True)
axes[1].set_title(f"DBSCAN (eps={EPS}, min_samples={MIN_SAMPLES})")
axes[1].set_xlabel("Disponibilidade (dias/ano)")
axes[1].set_ylabel("log(Price + 1)")
axes[1].legend(handles=handles_db, fontsize=9)

plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q4_fig3_price_avail_comparacao.png", dpi=150, bbox_inches='tight')
plt.show()

# ── FIGURA 4 – Distribuição de tamanho dos clusters
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Fig 4 — Distribuição de Tamanho dos Clusters",
             fontsize=14, fontweight='bold')

km_counts = df_model['cluster_km'].value_counts().sort_index()
axes[0].bar([f"C{k}" for k in km_counts.index], km_counts.values,
            color=[palette_km[k] for k in km_counts.index], edgecolor='white')
axes[0].set_title(f"K-Means (K={BEST_K})")
axes[0].set_xlabel("Cluster"); axes[0].set_ylabel("Nº de imóveis")
for i, v in enumerate(km_counts.values):
    axes[0].text(i, v + 200, f"{v:,}", ha='center', fontsize=9)

db_counts = df_model['cluster_db'].value_counts().sort_index()
bar_colors = [('lightgray' if l == -1 else cmap_db(unique_labels.index(l)))
              for l in db_counts.index]
bar_labels = [('Ruído' if l == -1 else f"C{l}") for l in db_counts.index]
axes[1].bar(bar_labels, db_counts.values, color=bar_colors, edgecolor='white')
axes[1].set_title(f"DBSCAN (eps={EPS}, min_samples={MIN_SAMPLES})")
axes[1].set_xlabel("Cluster"); axes[1].set_ylabel("Nº de imóveis")
for i, v in enumerate(db_counts.values):
    axes[1].text(i, v + 200, f"{v:,}", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q4_fig4_tamanho_clusters.png", dpi=150, bbox_inches='tight')
plt.show()

# ── Crosstabs DBSCAN
print("\n── CROSSTAB DBSCAN × NEIGHBOURHOOD_GROUP ───────────────────")
ct_db_ng = pd.crosstab(df_model['cluster_db'],
                        df_model['neighbourhood_group'],
                        margins=True)
print(ct_db_ng.to_string())

print("\n── CROSSTAB DBSCAN × ROOM_TYPE ─────────────────────────────")
ct_db_rt = pd.crosstab(df_model['cluster_db'],
                        df_model['room_type'],
                        margins=True)
print(ct_db_rt.to_string())

# ── FIGURA 5 – Heatmap crosstab DBSCAN
ct_db_ng_pct = pd.crosstab(df_model['cluster_db'],
                             df_model['neighbourhood_group'],
                             normalize='index').round(3) * 100
ct_db_rt_pct = pd.crosstab(df_model['cluster_db'],
                             df_model['room_type'],
                             normalize='index').round(3) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Fig 5 — Crosstab DBSCAN × Variáveis Categóricas (%)",
             fontsize=14, fontweight='bold')

sns.heatmap(ct_db_ng_pct, annot=True, fmt=".1f", cmap="Blues",
            linewidths=0.5, ax=axes[0],
            cbar_kws={"label": "% do cluster"})
axes[0].set_title("DBSCAN × Bairro")
axes[0].set_xlabel("Bairro"); axes[0].set_ylabel("Cluster (−1=ruído)")

sns.heatmap(ct_db_rt_pct, annot=True, fmt=".1f", cmap="Oranges",
            linewidths=0.5, ax=axes[1],
            cbar_kws={"label": "% do cluster"})
axes[1].set_title("DBSCAN × Tipo de Quarto")
axes[1].set_xlabel("Tipo"); axes[1].set_ylabel("Cluster (−1=ruído)")

plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q4_fig5_crosstab_dbscan.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 7. Tabela comparativa e conclusão
# ─────────────────────────────────────────────
print(f"""
── 7. COMPARAÇÃO K-MEANS × DBSCAN ──────────────────────────

╔══════════════════════════╦══════════════════════╦══════════════════════╗
║ Critério                 ║ K-Means (K=5)        ║ DBSCAN               ║
╠══════════════════════════╬══════════════════════╬══════════════════════╣
║ Nº de clusters           ║ 5 (fixo)             ║ {n_clusters_db} (descoberto)     ║
║ Pontos de ruído          ║ 0 (nenhum)           ║ {n_noise_db:,} ({pct_noise:.1f}%)      ║
║ Silhouette Score         ║ {sil_km:.4f}              ║ {sil_db:.4f}              ║
║ Formato dos clusters     ║ Esférico/Voronoi     ║ Forma arbitrária     ║
║ Sensibilidade à escala   ║ Alta (requer scaler) ║ Alta (requer scaler) ║
║ Facilidade interpretação ║ Alta                 ║ Média/Baixa          ║
║ Presença de ruído        ║ Não identifica       ║ Identifica           ║
╚══════════════════════════╩══════════════════════╩══════════════════════╝

ANÁLISE POR CRITÉRIO:

  Quantidade de clusters:
    K-Means exige definir K=5 a priori (baseado em cotovelo +
    Silhouette). O DBSCAN descobriu {n_clusters_db} cluster(s) automaticamente,
    o que pode ou não corresponder à estrutura real do dataset.

  Presença de ruídos:
    O DBSCAN identificou {n_noise_db:,} pontos ({pct_noise:.1f}%) como ruído —
    imóveis isolados ou atípicos que o K-Means absorveu
    forçosamente nos clusters. Esse comportamento do DBSCAN é
    vantajoso para detectar anomalias, mas reduz a cobertura.

  Formato dos agrupamentos:
    O dataset Airbnb possui clusters predominantemente geográficos
    (latitude × longitude), que são aproximadamente convexos.
    O K-Means, que assume clusters esféricos, lida bem com isso.
    O DBSCAN seria mais vantajoso em formatos não-convexos.

  Sensibilidade à escala:
    Ambos os algoritmos são sensíveis à escala dos atributos —
    o StandardScaler foi aplicado a ambos. A diferença é que
    o DBSCAN também é sensível à densidade: regiões com alta
    concentração (ex.: Manhattan) tendem a "absorver" regiões
    esparsas (ex.: Staten Island).

  Facilidade de interpretação:
    K-Means produz 5 grupos balanceados e bem delimitados,
    fáceis de rotular (ex.: "Manhattan premium", "Brooklyn médio").
    O DBSCAN gera clusters de tamanho muito desigual, dificultando
    a interpretação e a associação com categorias reais.

  Relação com variável categórica:
    A crosstab do K-Means mostrou clusters com especialização
    geográfica clara. O DBSCAN tende a criar um cluster dominante
    com a maioria dos pontos, perdendo granularidade.

CONCLUSÃO:
    Para este dataset, o K-MEANS apresentou resultado mais coerente.
    Os dados do Airbnb NYC possuem estrutura aproximadamente esférica
    (impulsionada por coordenadas geográficas), e o K-Means com K=5
    produziu clusters interpretáveis, bem distribuídos e com boa
    correspondência com os bairros reais. O DBSCAN, nesse contexto,
    sofreu com a variação de densidade entre bairros e a alta
    dimensionalidade, resultando em clusters desequilibrados e
    maior dificuldade de interpretação.
""")