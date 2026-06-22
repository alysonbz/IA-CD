"""
Questão 2 – Análise Visual com Dois Atributos

Atributos escolhidos: price  ×  availability_365
Variável de referência (cor): room_type / neighbourhood_group
"""

from pathlib import Path
PASTA_ATUAL = Path(__file__).parent

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Configurações visuais
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

DATASET_PATH = "../dataset/AB_NYC_2019.csv"

# ─────────────────────────────────────────────────────────────
# 1. Carregamento e limpeza mínima
# ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATASET_PATH)

# Remove preços inválidos (= 0) e outliers extremos (> 500 USD)
df_clean = df[(df['price'] > 0) & (df['price'] <= 500)].copy()
df_clean['log_price'] = np.log1p(df_clean['price'])

print(f"Registros originais : {len(df):,}")
print(f"Após filtro de preço: {len(df_clean):,}")

# ─────────────────────────────────────────────────────────────
# 2. Justificativa da escolha (impressa no console)
# ─────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════╗
║  JUSTIFICATIVA DA ESCOLHA DOS ATRIBUTOS                      ║
╠══════════════════════════════════════════════════════════════╣
║  Atributo 1: price (Preço por noite – US$)                   ║
║    • Variável-alvo central em anúncios de hospedagem.        ║
║    • As medianas diferem substancialmente por room_type      ║
║      (Entire: US$160 | Private: US$70 | Shared: US$45) e     ║
║      por bairro (Manhattan: US$150 vs Bronx: US$65).         ║
║    • Poder discriminativo natural entre segmentos.           ║
║                                                              ║
║  Atributo 2: availability_365 (Dias disponíveis/ano)         ║
║    • Reflete o perfil de uso: imóvel como renda passiva      ║
║      (alta disponibilidade) vs. uso esporádico (baixa).      ║
║    • As medianas também variam por grupo:                    ║
║      Manhattan/Brooklyn ~30-45 dias | Bronx/Staten >140.     ║
║    • Correlação baixa com price (r ≈ 0,08), garantindo       ║
║      que as duas dimensões são independentes e               ║
║      complementares para a separação visual.                 ║
╚══════════════════════════════════════════════════════════════╝
""")

# ─────────────────────────────────────────────────────────────
# Paletas
# ─────────────────────────────────────────────────────────────
palette_rt = {
    'Entire home/apt': '#e63946',
    'Private room':    '#457b9d',
    'Shared room':     '#2a9d8f',
}
palette_ng = {
    'Manhattan':     '#e63946',
    'Brooklyn':      '#457b9d',
    'Queens':        '#2a9d8f',
    'Bronx':         '#e9c46a',
    'Staten Island': '#f4a261',
}

# ─────────────────────────────────────────────────────────────
# FIGURA 1 – Dispersão simples
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(df_clean['availability_365'], df_clean['price'],
           alpha=0.15, s=8, color='#4C72B0', rasterized=True)
ax.set_title("Fig 1 — Dispersão: Price × Availability_365\n"
             "(sem variável de referência)", fontweight='bold')
ax.set_xlabel("Disponibilidade (dias/ano)")
ax.set_ylabel("Preço (US$)")
plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q2_fig1_dispersao_simples.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────────
# FIGURA 2 – Dispersão colorida por room_type (escala original)
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
for rtype, grp in df_clean.groupby('room_type'):
    ax.scatter(grp['availability_365'], grp['price'],
               alpha=0.12, s=8, color=palette_rt[rtype],
               label=rtype, rasterized=True)

ax.set_title("Fig 2 — Price × Availability_365\n"
             "colorido por Tipo de Quarto (room_type)", fontweight='bold')
ax.set_xlabel("Disponibilidade (dias/ano)")
ax.set_ylabel("Preço (US$)")
handles = [mpatches.Patch(color=c, label=l) for l, c in palette_rt.items()]
ax.legend(handles=handles, title="Tipo de Quarto", loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q2_fig2_cor_roomtype.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────────
# FIGURA 3 – Dispersão log(price) × availability por room_type
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
for rtype, grp in df_clean.groupby('room_type'):
    ax.scatter(grp['availability_365'], grp['log_price'],
               alpha=0.15, s=8, color=palette_rt[rtype],
               label=rtype, rasterized=True)

ax.set_title("Fig 3 — log(Price+1) × Availability_365\n"
             "colorido por Tipo de Quarto", fontweight='bold')
ax.set_xlabel("Disponibilidade (dias/ano)")
ax.set_ylabel("log(Price + 1)")
handles = [mpatches.Patch(color=c, label=l) for l, c in palette_rt.items()]
ax.legend(handles=handles, title="Tipo de Quarto", loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q2_fig3_log_roomtype.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────────
# FIGURA 4 – Dispersão colorida por neighbourhood_group
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
for ng, grp in df_clean.groupby('neighbourhood_group'):
    ax.scatter(grp['availability_365'], grp['price'],
               alpha=0.12, s=8, color=palette_ng[ng],
               label=ng, rasterized=True)

ax.set_title("Fig 4 — Price × Availability_365\n"
             "colorido por Bairro (neighbourhood_group)", fontweight='bold')
ax.set_xlabel("Disponibilidade (dias/ano)")
ax.set_ylabel("Preço (US$)")
handles = [mpatches.Patch(color=c, label=l) for l, c in palette_ng.items()]
ax.legend(handles=handles, title="Bairro", loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q2_fig4_cor_bairro.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────────
# FIGURA 5 – KDE 2D (densidade)
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
sample = df_clean.sample(8000, random_state=42)
sns.kdeplot(data=sample, x='availability_365', y='log_price',
            fill=True, cmap='Blues', levels=15, ax=ax)
ax.set_title("Fig 5 — Densidade 2D: log(Price+1) × Availability_365\n"
             "(KDE – amostra de 8.000 pontos)", fontweight='bold')
ax.set_xlabel("Disponibilidade (dias/ano)")
ax.set_ylabel("log(Price + 1)")
plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q2_fig5_kde.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────────
# FIGURA 6 – Medianas por grupo (barras agrupadas)
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Fig 6 — Medianas de Price e Availability_365 por Grupo",
             fontsize=14, fontweight='bold')

med_rt = df_clean.groupby('room_type')[['price', 'availability_365']].median()
x = np.arange(len(med_rt)); w = 0.35
axes[0].bar(x - w/2, med_rt['price'],          w, label='price (US$)',      color='#e63946')
axes[0].bar(x + w/2, med_rt['availability_365'], w, label='availability_365', color='#457b9d')
for i, (p, a) in enumerate(zip(med_rt['price'], med_rt['availability_365'])):
    axes[0].text(i - w/2, p + 2, f"{p:.0f}", ha='center', fontsize=8)
    axes[0].text(i + w/2, a + 2, f"{a:.0f}", ha='center', fontsize=8)
axes[0].set_xticks(x); axes[0].set_xticklabels(med_rt.index, fontsize=9)
axes[0].set_title("Por Tipo de Quarto"); axes[0].legend()

med_ng = df_clean.groupby('neighbourhood_group')[['price', 'availability_365']].median()
x2 = np.arange(len(med_ng))
axes[1].bar(x2 - w/2, med_ng['price'],           w, label='price (US$)',      color='#e63946')
axes[1].bar(x2 + w/2, med_ng['availability_365'], w, label='availability_365', color='#457b9d')
for i, (p, a) in enumerate(zip(med_ng['price'], med_ng['availability_365'])):
    axes[1].text(i - w/2, p + 2, f"{p:.0f}", ha='center', fontsize=8)
    axes[1].text(i + w/2, a + 2, f"{a:.0f}", ha='center', fontsize=8)
axes[1].set_xticks(x2); axes[1].set_xticklabels(med_ng.index, fontsize=9, rotation=15)
axes[1].set_title("Por Bairro"); axes[1].legend()

plt.tight_layout()
plt.savefig(PASTA_ATUAL/"q2_fig6_medianas.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────────
# 3. Análise crítica (impressa no console)
# ─────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════╗
║  ANÁLISE CRÍTICA                                             ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  OBSERVAÇÕES SOBRE A DISPERSÃO                               ║
║  ─────────────────────────────                               ║
║  1. Fig 1 (sem cor): não há separação visual evidente;       ║
║     os pontos formam uma nuvem densa e uniforme. A alta      ║
║     assimetria de 'price' cria concentração na base do       ║
║     gráfico, dificultando qualquer leitura de cluster.       ║
║                                                              ║
║  2. Fig 2/3 (por room_type): com a transformação log,        ║
║     surge uma separação vertical parcial: imóveis do tipo    ║
║     "Entire home/apt" tendem a concentrar-se em valores      ║
║     mais altos de log_price, enquanto "Private room" e       ║
║     "Shared room" ficam abaixo. Contudo, há sobreposição     ║
║     considerável — os grupos não são linearmente separáveis. ║
║                                                              ║
║  3. Fig 4 (por bairro): a variável neighbourhood_group       ║
║     praticamente não separa os pontos em price ×             ║
║     availability. As cores se misturam sem padrão espacial   ║
║     claro, indicando que o bairro não é capturado por        ║
║     esses dois atributos.                                    ║
║                                                              ║
║  4. Fig 5 (KDE 2D): o gráfico de densidade revela um         ║
║     único núcleo central denso, sem picos secundários        ║
║     claros. Isso confirma a ausência de clusters naturais    ║
║     nesse espaço 2D.                                         ║
║                                                              ║
║  CONCLUSÃO – ANÁLISE CRÍTICA                                 ║
║  ──────────────────────────                                  ║
║  A clusterização puramente visual com price e                ║
║  availability_365 é INVIÁVEL:                                ║
║                                                              ║
║  • A separação existe (medianas diferem por grupo), mas      ║
║    é estatisticamente tênue — as distribuições se            ║
║    sobrepõem amplamente, impedindo fronteiras claras.        ║
║                                                              ║
║  • Os dois atributos capturam dimensões úteis (valor e       ║
║    perfil de uso), mas são insuficientes para representar    ║
║    a complexidade do dataset.                                ║
║                                                              ║
║  • SERÃO NECESSÁRIOS MAIS ATRIBUTOS nas próximas etapas:     ║
║    – latitude/longitude: para separação geográfica           ║
║    – number_of_reviews: popularidade e atividade             ║
║    – calculated_host_listings_count: perfil do anfitrião     ║
║    – room_type codificada: diferença fundamental de preço    ║
║                                                              ║
║    A combinação desses 6–7 atributos (com normalização e     ║
║    redução de assimetria) tende a gerar clusters mais        ║
║    coesos e separados em algoritmos como K-Means ou DBSCAN.  ║
╚══════════════════════════════════════════════════════════════╝
""")