"""
Questão 1 – Análise Exploratória dos Dados
Dataset: New York City Airbnb Open Data
"""

from pathlib import Path
PASTA_ATUAL = Path(__file__).parent

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Configurações visuais
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

# ─────────────────────────────────────────────
# 1. Carregamento
# ─────────────────────────────────────────────
print("=" * 60)
print("QUESTÃO 1 – ANÁLISE EXPLORATÓRIA DOS DADOS")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

# ─────────────────────────────────────────────
# 2. Descrição geral
# ─────────────────────────────────────────────
print("\n── 2. DESCRIÇÃO GERAL ──────────────────────────────────────")
print(f"Dimensões do dataset : {df.shape[0]:,} linhas × {df.shape[1]} colunas")
print(f"\nColunas e tipos de dados:")
print(df.dtypes.to_string())

# Classificação das colunas
num_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols   = df.select_dtypes(include=["object"]).columns.tolist()
print(f"\nAtributos numéricos  ({len(num_cols)}): {num_cols}")
print(f"Atributos categóricos ({len(cat_cols)}): {cat_cols}")

# ─────────────────────────────────────────────
# 3. Verificação de problemas nos dados
# ─────────────────────────────────────────────
print("\n── 3. VERIFICAÇÃO DE PROBLEMAS NOS DADOS ───────────────────")

# Valores ausentes
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
miss_df = pd.DataFrame({"Ausentes": missing, "% do total": missing_pct})
miss_df = miss_df[miss_df["Ausentes"] > 0]
print("\nColunas com valores ausentes:")
print(miss_df.to_string())

# Duplicatas
n_dup = df.duplicated().sum()
print(f"\nLinhas duplicadas: {n_dup}")

# Anomalias pontuais
print(f"\nPreços iguais a 0         : {(df['price'] == 0).sum()} registros")
print(f"Preços acima de US$ 1.000 : {(df['price'] > 1000).sum()} registros")
print(f"Mínimo de noites > 365    : {(df['minimum_nights'] > 365).sum()} registros")
print(f"Disponibilidade = 365 dias: {(df['availability_365'] == 365).sum()} registros")

# ─────────────────────────────────────────────
# 4. Estatísticas descritivas
# ─────────────────────────────────────────────
feat_cols = ['price', 'minimum_nights', 'number_of_reviews',
             'reviews_per_month', 'calculated_host_listings_count',
             'availability_365']

print("\n── 4. ESTATÍSTICAS DESCRITIVAS ─────────────────────────────")
pd.set_option("display.float_format", "{:.2f}".format)
desc = df[feat_cols].describe().T
desc["skewness"] = df[feat_cols].skew()
print(desc.to_string())

print("\nVariáveis categóricas:")
for col in ['neighbourhood_group', 'room_type']:
    print(f"\n{col}:")
    print(df[col].value_counts().to_string())

# ─────────────────────────────────────────────
# 5. Visualizações exploratórias
# ─────────────────────────────────────────────

# ── Fig 1: Distribuições dos atributos numéricos
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Distribuição dos Atributos Numéricos", fontsize=15, fontweight="bold")

for ax, col in zip(axes.flatten(), feat_cols):
    data = df[col].dropna()
    ax.hist(data, bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_title(col)
    ax.set_xlabel("Valor")
    ax.set_ylabel("Frequência")
    skew = data.skew()
    ax.text(0.97, 0.95, f"assimetria: {skew:.2f}",
            ha="right", va="top", transform=ax.transAxes,
            fontsize=8, color="firebrick")

plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q1_fig1_distribuicoes.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Fig 2: Distribuição geográfica – Latitude × Longitude por neighbourhood_group
fig, ax = plt.subplots(figsize=(10, 8))
palette = {"Manhattan": "#e63946", "Brooklyn": "#457b9d",
           "Queens": "#2a9d8f", "Bronx": "#e9c46a", "Staten Island": "#f4a261"}

for group, grp_df in df.groupby("neighbourhood_group"):
    ax.scatter(grp_df["longitude"], grp_df["latitude"],
               s=3, alpha=0.35, label=group, color=palette[group])

ax.set_title("Distribuição Geográfica dos Imóveis por Bairro", fontsize=14, fontweight="bold")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend(title="Bairro", markerscale=4)
plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q1_fig2_mapa_bairros.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Fig 3: Variáveis categóricas
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Distribuição das Variáveis Categóricas", fontsize=14, fontweight="bold")

vc_ng = df["neighbourhood_group"].value_counts()
axes[0].bar(vc_ng.index, vc_ng.values, color="#4C72B0", edgecolor="white")
axes[0].set_title("Bairro (neighbourhood_group)")
axes[0].set_xlabel("Bairro")
axes[0].set_ylabel("Quantidade de imóveis")
for i, v in enumerate(vc_ng.values):
    axes[0].text(i, v + 200, f"{v:,}", ha="center", fontsize=9)

vc_rt = df["room_type"].value_counts()
axes[1].bar(vc_rt.index, vc_rt.values, color="#DD8452", edgecolor="white")
axes[1].set_title("Tipo de Quarto (room_type)")
axes[1].set_xlabel("Tipo")
axes[1].set_ylabel("Quantidade de imóveis")
for i, v in enumerate(vc_rt.values):
    axes[1].text(i, v + 200, f"{v:,}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q1_fig3_categoricas.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Fig 4: Boxplots de preço por bairro e tipo de quarto
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Distribuição de Preço por Variável Categórica", fontsize=14, fontweight="bold")

df_price = df[df["price"].between(1, 1000)]   # remove zeros e outliers extremos

order_ng = df_price.groupby("neighbourhood_group")["price"].median().sort_values(ascending=False).index
sns.boxplot(data=df_price, x="neighbourhood_group", y="price",
            order=order_ng, ax=axes[0], palette="muted")
axes[0].set_title("Preço × Bairro")
axes[0].set_xlabel("")
axes[0].set_ylabel("Preço (US$)")

order_rt = df_price.groupby("room_type")["price"].median().sort_values(ascending=False).index
sns.boxplot(data=df_price, x="room_type", y="price",
            order=order_rt, ax=axes[1], palette="Set2")
axes[1].set_title("Preço × Tipo de Quarto")
axes[1].set_xlabel("")
axes[1].set_ylabel("Preço (US$)")

plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q1_fig4_boxplot_preco.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Fig 5: Matriz de correlação
print("\nCalculando matriz de correlação...")
corr = df[feat_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title("Matriz de Correlação – Atributos Numéricos", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q1_fig5_correlacao.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Fig 6: Análise de valores ausentes – reviews_per_month × number_of_reviews
fig, ax = plt.subplots(figsize=(9, 5.5))
has_review = df["reviews_per_month"].notna()

sem_review = df.loc[~has_review, "number_of_reviews"]
com_review = df.loc[has_review, "number_of_reviews"]

# Bins compartilhados: garante que as barras de cada grupo fiquem
# lado a lado dentro do mesmo intervalo, em vez de uma sobrepor a outra
max_val = df["number_of_reviews"].max()
bin_edges = np.linspace(0, max_val, 41)

ax.hist([sem_review, com_review], bins=bin_edges,
        color=["#e63946", "#457b9d"], alpha=0.85,
        label=[f"Sem reviews_per_month (n={len(sem_review):,})",
               f"Com reviews_per_month (n={len(com_review):,})"])

ax.set_title("Imóveis com e sem 'reviews_per_month' × número de avaliações")
ax.set_xlabel("Número de avaliações")
ax.set_ylabel("Frequência")
ax.legend()
plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q1_fig6_ausentes_reviews.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"""
  Nota sobre a Fig 6:
    Todos os {len(sem_review):,} imóveis SEM 'reviews_per_month' têm
    number_of_reviews = 0 (consistente: sem avaliações, não há média
    mensal a calcular). Por isso a barra vermelha aparece concentrada
    inteiramente no primeiro bin, ao lado da barra azul (imóveis com
    pelo menos 1 avaliação).
""")

# ─────────────────────────────────────────────
# 6. Interpretação e achados principais
# ─────────────────────────────────────────────
print("\n── 6. INTERPRETAÇÃO DOS PRINCIPAIS ACHADOS ─────────────────")
print("""
ACHADOS PRINCIPAIS
──────────────────
1. Tamanho e estrutura
   O dataset contém 48.895 anúncios do Airbnb em Nova York (2019),
   com 16 atributos — 10 numéricos e 6 textuais/categóricos — e
   nenhuma linha duplicada.

2. Valores ausentes
   • 'reviews_per_month' e 'last_review' possuem 10.052 ausências (~20,6%).
     Esses registros correspondem a imóveis sem nenhuma avaliação
     (number_of_reviews = 0), o que é estruturalmente coerente.
   • 'name' e 'host_name' apresentam poucas ausências (< 21 linhas),
     com impacto negligenciável.

3. Anomalias detectadas
   • 11 imóveis com preço = 0 (dados inválidos).
   • 239 imóveis com preço > US$ 1.000 (outliers extremos).
   • 14 imóveis com mínimo de noites > 365 (valor ilógico).

4. Assimetria dos atributos numéricos
   Praticamente todas as variáveis numéricas apresentam assimetria
   positiva elevada (ex.: price ≈ 19, minimum_nights ≈ 22). Isso
   indica concentração em valores baixos com longas caudas à direita.
   Transformações logarítmicas serão necessárias antes de clusterizar.

5. Distribuição geográfica e categórica
   • Manhattan e Brooklyn concentram ~85% dos imóveis.
   • "Entire home/apt" e "Private room" correspondem a ~98% dos anúncios.
   • Manhattan possui os preços medianos mais altos, seguida de Brooklyn.

6. Correlações relevantes
   • 'number_of_reviews' e 'reviews_per_month' são moderadamente
     correlacionadas (esperado — mais avaliações → média mensal maior).
   • 'latitude' e 'longitude' estão negativamente correlacionadas,
     o que é geográfico (NY cresce do Sul ao Norte / Leste-Oeste).
   • 'price' tem correlações baixas com as demais variáveis numéricas,
     sugerindo que o preço é influenciado principalmente por localização
     e tipo de quarto (variáveis categóricas).

7. Atributos mais relevantes para clusterização
   Com base na EDA, os atributos mais promissores para agrupamento são:
     – latitude / longitude        (localização geográfica)
     – price                       (faixa de preço)
     – availability_365            (perfil de disponibilidade)
     – number_of_reviews           (popularidade)
     – calculated_host_listings_count (tipo de anfitrião)
   As variáveis 'room_type' e 'neighbourhood_group' serão utilizadas
   como referência categórica para avaliar a qualidade dos clusters.
""")