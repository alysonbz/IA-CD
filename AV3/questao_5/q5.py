"""
Questão 5 – Impacto da Normalização na Clusterização

Cenários comparados:
    1. Sem normalização (dados brutos)
    2. Z-score  (StandardScaler)
    3. Min-Max  (MinMaxScaler)
    4. Log + Z-score (transformação logarítmica + padronização)

Mesmo conjunto de atributos da Q3/Q4:
    latitude, longitude, price*, availability_365,
    number_of_reviews*, calculated_host_listings_count
    (* versão log na etapa 4)
Variável de referência: room_type / neighbourhood_group
"""

from pathlib import Path
PASTA_ATUAL = Path(__file__).parent

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Configurações visuais (mesmo padrão Q1–Q4)
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
K_RANGE      = range(2, 11)

# ─────────────────────────────────────────────
print("=" * 60)
print("QUESTÃO 5 – IMPACTO DA NORMALIZAÇÃO NA CLUSTERIZAÇÃO")
print("=" * 60)

# ─────────────────────────────────────────────
# Carregamento e limpeza (idêntico Q3/Q4)
# ─────────────────────────────────────────────
df = pd.read_csv(DATASET_PATH)
df_clean = df[(df['price'] > 0) & (df['price'] <= 1000)].copy()
df_clean = df_clean[df_clean['minimum_nights'] <= 365].copy()
df_clean['reviews_per_month'] = df_clean['reviews_per_month'].fillna(0)
df_clean.reset_index(drop=True, inplace=True)

# Atributos BASE (sem transformação logarítmica)
FEATURES_RAW = ['latitude', 'longitude', 'price',
                'availability_365', 'number_of_reviews',
                'calculated_host_listings_count']

# Atributos com log (cenário 4)
df_clean['log_price']   = np.log1p(df_clean['price'])
df_clean['log_reviews'] = np.log1p(df_clean['number_of_reviews'])
FEATURES_LOG = ['latitude', 'longitude', 'log_price',
                'availability_365', 'log_reviews',
                'calculated_host_listings_count']

X_raw = df_clean[FEATURES_RAW].dropna()
df_model = df_clean.loc[X_raw.index].copy()
X_log = df_model[FEATURES_LOG].values

print(f"Registros utilizados : {len(X_raw):,}")
print(f"Atributos base       : {FEATURES_RAW}")

# ─────────────────────────────────────────────
# Definição dos 4 cenários
# ─────────────────────────────────────────────
scenarios = {
    "1. Sem normalização":    X_raw.values,
    "2. Z-score":             StandardScaler().fit_transform(X_raw),
    "3. Min-Max":             MinMaxScaler().fit_transform(X_raw),
    "4. Log + Z-score":       StandardScaler().fit_transform(X_log),
}

print("""
── CENÁRIOS DE NORMALIZAÇÃO ────────────────────────────────

  Cenário 1 – Sem normalização (dados brutos)
    Atributos na escala original. 'price' (0–1000) e
    'number_of_reviews' (0–629) dominam a métrica de distância,
    tornando latitude/longitude praticamente irrelevantes.

  Cenário 2 – Z-score (StandardScaler)
    Cada atributo com média=0 e desvio padrão=1.
    Equilibra escalas sem pressupor limites mínimo/máximo.
    Sensível a outliers (não os elimina, apenas reescala).

  Cenário 3 – Min-Max (MinMaxScaler)
    Cada atributo reescalado para [0, 1].
    Mantém a forma da distribuição original, mas comprime
    outliers extremos junto com os demais valores.

  Cenário 4 – Log + Z-score
    Primeiro aplica log(x+1) aos atributos com forte assimetria
    (price e number_of_reviews), corrigindo a cauda direita.
    Depois aplica Z-score para equalizar escalas.
    Estratégia adotada nas questões Q3 e Q4.
""")

# ─────────────────────────────────────────────
# Amostra para Silhouette (custo O(n²))
# ─────────────────────────────────────────────
np.random.seed(RANDOM_STATE)
N_SAMPLE  = 10_000
samp_idx  = np.random.choice(len(X_raw), size=N_SAMPLE, replace=False)

# ─────────────────────────────────────────────
# Loop: K-Means para cada cenário × cada K
# ─────────────────────────────────────────────
print("── CALCULANDO K-MEANS PARA TODOS OS CENÁRIOS ──────────────")
results = {}   # {cenário: {"inertias": [], "silhouettes": [], "best_k": int, "labels": array}}

for name, X_sc in scenarios.items():
    inertias, silhouettes = [], []
    X_samp = X_sc[samp_idx]
    for k in K_RANGE:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10,
                    random_state=RANDOM_STATE)
        km.fit(X_sc)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_samp, km.labels_[samp_idx],
                               random_state=RANDOM_STATE)
        silhouettes.append(sil)

    best_k   = list(K_RANGE)[int(np.argmax(silhouettes))]
    km_best  = KMeans(n_clusters=best_k, init='k-means++', n_init=15,
                      random_state=RANDOM_STATE)
    km_best.fit(X_sc)

    results[name] = {
        "inertias":    inertias,
        "silhouettes": silhouettes,
        "best_k":      best_k,
        "best_sil":    max(silhouettes),
        "labels":      km_best.labels_,
        "X_scaled":    X_sc,
    }
    print(f"  {name:30s} | melhor K={best_k} | Silhouette={max(silhouettes):.4f}")

# ─────────────────────────────────────────────
# FIGURA 1 – Cotovelo comparativo (4 cenários)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Fig 1 — Método do Cotovelo por Cenário de Normalização",
             fontsize=15, fontweight='bold')

colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
ks     = list(K_RANGE)

for ax, (name, res), color in zip(axes.flatten(), results.items(), colors):
    ax.plot(ks, res['inertias'], marker='o', color=color,
            linewidth=2, markersize=6)
    ax.fill_between(ks, res['inertias'], alpha=0.07, color=color)
    ax.axvline(res['best_k'], color='firebrick', linestyle='--', alpha=0.7,
               label=f"K={res['best_k']} (melhor Sil.)")
    ax.set_title(name)
    ax.set_xlabel("K")
    ax.set_ylabel("Inércia")
    ax.set_xticks(ks)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q5_fig1_cotovelo_cenarios.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# FIGURA 2 – Silhouette Score comparativo
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Fig 2 — Silhouette Score por Cenário de Normalização",
             fontsize=15, fontweight='bold')

for ax, (name, res), color in zip(axes.flatten(), results.items(), colors):
    ax.plot(ks, res['silhouettes'], marker='s', color=color,
            linewidth=2, markersize=6)
    ax.fill_between(ks, res['silhouettes'], alpha=0.07, color=color)
    ax.axvline(res['best_k'], color='firebrick', linestyle='--', alpha=0.7,
               label=f"Melhor K={res['best_k']} ({res['best_sil']:.3f})")
    ax.set_title(name)
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette Score")
    ax.set_xticks(ks)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q5_fig2_silhouette_cenarios.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# FIGURA 3 – Resumo comparativo (barras)
# ─────────────────────────────────────────────
scenario_names  = [n.split(". ", 1)[1] if ". " in n else n for n in results.keys()]
best_ks         = [res['best_k']   for res in results.values()]
best_sils       = [res['best_sil'] for res in results.values()]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Fig 3 — Resumo: Melhor K e Silhouette Score por Cenário",
             fontsize=14, fontweight='bold')

bars0 = axes[0].bar(scenario_names, best_ks, color=colors, edgecolor='white')
axes[0].set_title("Melhor K por Cenário")
axes[0].set_ylabel("K escolhido")
axes[0].set_ylim(0, max(best_ks) + 2)
for bar, v in zip(bars0, best_ks):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.1,
                 str(v), ha='center', fontsize=11, fontweight='bold')

bars1 = axes[1].bar(scenario_names, best_sils, color=colors, edgecolor='white')
axes[1].set_title("Melhor Silhouette Score por Cenário")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_ylim(0, max(best_sils) + 0.05)
for bar, v in zip(bars1, best_sils):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.002,
                 f"{v:.4f}", ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q5_fig3_resumo_comparativo.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# FIGURA 4 – Dispersão: log_price × availability
#            (clusters do melhor K de cada cenário)
# ─────────────────────────────────────────────
palette_list = [
    ['#e63946','#457b9d','#2a9d8f','#e9c46a','#f4a261',
     '#8ecae6','#219ebc','#023047','#ffb703','#fb8500'],
] * 4

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Fig 4 — Dispersão log(Price+1) × Availability_365\n"
             "colorido pelos clusters de cada cenário",
             fontsize=15, fontweight='bold')

log_price_vals = df_model['log_price'].values
avail_vals     = df_model['availability_365'].values

for ax, (name, res), pal in zip(axes.flatten(), results.items(), palette_list):
    labels = res['labels']
    n_cl   = res['best_k']
    for cl in range(n_cl):
        mask = labels == cl
        ax.scatter(avail_vals[mask], log_price_vals[mask],
                   s=4, alpha=0.25, color=pal[cl],
                   label=f"C{cl}", rasterized=True)
    ax.set_title(f"{name}\n(K={n_cl}, Sil={res['best_sil']:.3f})")
    ax.set_xlabel("Disponibilidade (dias/ano)")
    ax.set_ylabel("log(Price + 1)")
    handles = [mpatches.Patch(color=pal[c], label=f"C{c}") for c in range(n_cl)]
    ax.legend(handles=handles, fontsize=8, ncol=2)

plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q5_fig4_dispersao_cenarios.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# FIGURA 5 – Mapa geográfico por cenário
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Fig 5 — Mapa Geográfico dos Clusters por Cenário",
             fontsize=15, fontweight='bold')

lon_vals = df_model['longitude'].values
lat_vals = df_model['latitude'].values

for ax, (name, res), pal in zip(axes.flatten(), results.items(), palette_list):
    labels = res['labels']
    n_cl   = res['best_k']
    for cl in range(n_cl):
        mask = labels == cl
        ax.scatter(lon_vals[mask], lat_vals[mask],
                   s=3, alpha=0.3, color=pal[cl],
                   label=f"C{cl}", rasterized=True)
    ax.set_title(f"{name} (K={n_cl})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    handles = [mpatches.Patch(color=pal[c], label=f"C{c}") for c in range(n_cl)]
    ax.legend(handles=handles, fontsize=8, markerscale=3, ncol=2)

plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q5_fig5_mapa_cenarios.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# Crosstabs por cenário × neighbourhood_group
# ─────────────────────────────────────────────
print("\n── CROSSTABS: CLUSTER × NEIGHBOURHOOD_GROUP (por cenário) ──")
for name, res in results.items():
    df_model[f'cluster_{name[:1]}'] = res['labels']
    ct = pd.crosstab(res['labels'],
                     df_model['neighbourhood_group'],
                     normalize='index').round(3) * 100
    print(f"\n{name} (K={res['best_k']}):")
    print(ct.to_string())

# ─────────────────────────────────────────────
# FIGURA 6 – Heatmaps crosstab dos 4 cenários
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Fig 6 — Crosstab Cluster × Bairro (%) por Cenário de Normalização",
             fontsize=15, fontweight='bold')

cmaps = ['Blues', 'Oranges', 'Greens', 'Reds']
for ax, (name, res), cmap in zip(axes.flatten(), results.items(), cmaps):
    ct_pct = pd.crosstab(res['labels'],
                          df_model['neighbourhood_group'],
                          normalize='index').round(3) * 100
    sns.heatmap(ct_pct, annot=True, fmt=".1f", cmap=cmap,
                linewidths=0.4, ax=ax,
                cbar_kws={"label": "% do cluster"})
    ax.set_title(f"{name} (K={res['best_k']})")
    ax.set_xlabel("Bairro")
    ax.set_ylabel("Cluster")

plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q5_fig6_crosstab_cenarios.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# Tabela-resumo final
# ─────────────────────────────────────────────
print("\n── TABELA-RESUMO FINAL ─────────────────────────────────────")
summary = pd.DataFrame({
    "Cenário":          list(results.keys()),
    "Melhor K":         [res['best_k']   for res in results.values()],
    "Silhouette Score": [round(res['best_sil'], 4) for res in results.values()],
})
print(summary.to_string(index=False))

# ─────────────────────────────────────────────
# Conclusão
# ─────────────────────────────────────────────
best_scenario  = max(results, key=lambda n: results[n]['best_sil'])
worst_scenario = min(results, key=lambda n: results[n]['best_sil'])

sil_sem_norm = results["1. Sem normalização"]['best_sil']
sil_zscore   = results["2. Z-score"]['best_sil']
sil_minmax   = results["3. Min-Max"]['best_sil']
sil_logz     = results["4. Log + Z-score"]['best_sil']

k_sem_norm = results["1. Sem normalização"]['best_k']
k_zscore   = results["2. Z-score"]['best_k']
k_minmax   = results["3. Min-Max"]['best_k']
k_logz     = results["4. Log + Z-score"]['best_k']

print(f"""
── CONCLUSÃO: A NORMALIZAÇÃO AJUDOU? ───────────────────────

  RESULTADOS OBTIDOS:
    1. Sem normalização  | K={k_sem_norm}  | Silhouette={sil_sem_norm:.4f}
    2. Z-score           | K={k_zscore}  | Silhouette={sil_zscore:.4f}
    3. Min-Max            | K={k_minmax}  | Silhouette={sil_minmax:.4f}
    4. Log + Z-score      | K={k_logz}  | Silhouette={sil_logz:.4f}

  Maior Silhouette: {best_scenario} ({results[best_scenario]['best_sil']:.4f})
  Menor Silhouette: {worst_scenario} ({results[worst_scenario]['best_sil']:.4f})

  RESPOSTA: Pelo Silhouette Score puro, a normalização NÃO
  ajudou — pelo contrário, os dois melhores resultados vieram
  dos cenários SEM correção de assimetria (sem normalização e
  Min-Max), enquanto Z-score e Log+Z-score tiveram desempenho
  inferior. Esse resultado é contraintuitivo e exige explicação.

  POR QUE "SEM NORMALIZAÇÃO" E "MIN-MAX" TÊM SILHOUETTE ALTO?

  O Silhouette Score mede o quão bem separados e compactos são
  os clusters — mas não garante que essa separação seja
  significativa. Nos cenários 1 e 3:
    • 'price' (0–1000) e 'number_of_reviews' (0–629) dominam
      a métrica de distância Euclidiana, pois suas escalas são
      muito maiores que as de latitude/longitude (~0,3 grau de
      variação) e calculated_host_listings_count.
    • O K-Means encontra poucos clusters (K={k_sem_norm} e K={k_minmax})
      separados quase exclusivamente por faixa de preço/reviews,
      ignorando a estrutura geográfica. Poucos clusters grandes
      e dominados por 1–2 atributos tendem a gerar Silhouette
      artificialmente alto, pois a separação nessas variáveis
      de escala grande é muito nítida.
    • Ou seja, o Silhouette alto aqui reflete "fácil separação
      em 1 dimensão dominante", não uma estrutura multivariada
      rica nos dados.

  POR QUE Z-SCORE E LOG+Z-SCORE TÊM SILHOUETTE MENOR?

    • Ao equalizar as escalas, latitude/longitude passam a
      pesar tanto quanto price/reviews. Isso introduz mais
      dimensões relevantes na distância, tornando os clusters
      menos "limpos" geometricamente (Silhouette menor), mas
      também mais REPRESENTATIVOS da estrutura real do dataset
      (geografia + preço + popularidade + perfil do anfitrião).
    • O Z-score (cenário 2) chegou a K={k_zscore}, coerente com as
      5 regiões geográficas de NY usadas como referência
      categórica — diferente dos cenários 1 e 3, que colapsam
      em poucos clusters dominados por preço.
    • O Log+Z-score (cenário 4) teve o menor Silhouette
      ({sil_logz:.4f}) e K={k_logz}, indicando que a correção da
      assimetria de price/reviews tornou os clusters ainda
      menos separáveis nessas dimensões — o que é esperado,
      já que o objetivo do log é justamente reduzir a influência
      desproporcional desses outliers.

  MUDANÇAS OBSERVADAS:
    • Melhor K: variou de {min(k_sem_norm, k_zscore, k_minmax, k_logz)} a
      {max(k_sem_norm, k_zscore, k_minmax, k_logz)} entre os cenários — não há
      convergência para um único K, ao contrário do esperado.
      Apenas o Z-score produziu K=5, alinhado à Q3.
    • Silhouette Score: caiu do cenário 1 (sem normalização)
      para os cenários 2 e 4 — comportamento oposto ao que se
      esperaria se a normalização "limpasse" os clusters.
    • Distribuição dos clusters: nos cenários 1 e 3, os grupos
      tendem a ser poucos e grandes (dominados por price);
      no Z-score, mais clusters e mais equilibrados.
    • Crosstab: precisa ser inspecionada visualmente (Fig 6)
      para confirmar se o Z-score, apesar do Silhouette menor,
      produz correspondência mais clara com neighbourhood_group.

  VEREDICTO FINAL:
    Neste dataset, o Silhouette Score isoladamente é uma
    métrica ENGANOSA para decidir se a normalização "ajudou":
    cenários sem correção de escala (1 e 3) vencem no Silhouette
    justamente porque colapsam a clusterização em torno das
    variáveis de maior magnitude (price, number_of_reviews),
    não porque encontram uma estrutura mais rica.

    A normalização (especialmente Z-score) PREJUDICOU o
    Silhouette Score bruto, mas é necessária para que latitude,
    longitude e os demais atributos tenham peso comparável na
    distância — sem ela, o K-Means na prática ignora a geografia.
    A escolha do cenário ideal depende do objetivo: se o
    objetivo é maximizar separação estatística pura, os
    cenários 1/3 "vencem"; se o objetivo é obter clusters
    interpretáveis e multivariados (como buscado nas Q3/Q4),
    a normalização via Z-score (ou Log+Z-score) é preferível,
    mesmo com Silhouette mais baixo.
""")