###################################################################################################################
#----------------------------------CLASSIFICAÇÃO-------------------------------------------------------------------
###################################################################################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# ── 1. Carregar e explorar o dataset ────────────────────────────────────────
colunas = [
    "status_conta", "duracao_meses", "historico_credito", "proposito",
    "valor_credito", "poupanca", "emprego_atual", "taxa_parcelamento",
    "estado_civil", "outros_devedores", "residencia_atual", "propriedade",
    "idade", "outros_planos", "habitacao", "creditos_banco",
    "tipo_emprego", "dependentes", "telefone", "trabalhador_estrangeiro",
    "classe",
]
df1 = pd.read_csv("statlog\statlog_german_credit_data.csv", header=None, names=colunas)
print(df1.shape)
print(df1.head())
print(df1.describe())

# ── 2. Tratar valores ausentes e inconsistentes ──────────────────────────────
print(df1.isnull().sum())  # sem valores nulos neste dataset

# ── 3. Analisar atributos mais relevantes ────────────────────────────────────
# Codificação e ajuste da classe
for col in df1.select_dtypes(include=["object", "string"]).columns:
    df1[col] = LabelEncoder().fit_transform(df1[col])

df1["classe"] = df1["classe"] - 1  # 1→0 (bom), 2→1 (mau) 

# ── Gráfico 1: Distribuição da Classe ────────────────────────────────────────
contagens = df1["classe"].value_counts().sort_index()
fig_dist, ax_dist = plt.subplots(figsize=(6, 5))
bars = ax_dist.bar(["Bom pagador (0)", "Mau pagador (1)"], contagens.values,
                   color=["steelblue", "orange"], edgecolor="white", linewidth=1.5)
ax_dist.bar_label(bars, padding=4, fontsize=11, fontweight="bold")
ax_dist.set_title("Distribuição da Classe Alvo", fontsize=13, fontweight="bold")
ax_dist.set_ylabel("Quantidade")
ax_dist.set_ylim(0, max(contagens.values) * 1.15)
ax_dist.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
fig_dist.savefig("distribuicao_classe.png", dpi=150, bbox_inches="tight")
plt.show()

# ---- Correlação ---------------------------------------------------------------
correlacoes = df1.drop("classe", axis=1).corrwith(df1["classe"]).abs().sort_values(ascending=False)
print(correlacoes)

# ── Gráfico 2: Barras de Correlação ──────────────────────────────────────────
fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
cores_bar = ["steelblue" if i < 5 else "lightsteelblue" for i in range(len(correlacoes))]
ax_bar.barh(correlacoes.index[::-1], correlacoes.values[::-1],
            color=cores_bar[::-1], edgecolor="white")
ax_bar.set_title("Correlação dos Atributos com a Classe", fontsize=13, fontweight="bold")
ax_bar.set_xlabel("|Correlação de Pearson|")
ax_bar.axvline(0.1, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="referência 0.10")
ax_bar.legend()
ax_bar.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
fig_bar.savefig("correlacoes_barras.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Matriz de Correlação ─────────────────────────────────────────────────────
# Após o LabelEncoder, inverte o status_conta para refletir o risco corretamente
df1["status_conta"] = df1["status_conta"].max() - df1["status_conta"]
fig_corr, ax_corr = plt.subplots(figsize=(10, 7))
top_feats = list(correlacoes.head(8).index) + ["classe"]
corr_mat  = df1[top_feats].corr()
sns.heatmap(corr_mat, ax=ax_corr, cmap="RdBu_r", center=0,
            annot=True, fmt=".2f", linewidths=0.5,
            annot_kws={"size": 9}, cbar_kws={"shrink": 0.8})
ax_corr.set_title("Matriz de Correlação — 8 Atributos + Classe", fontsize=13, fontweight="bold")
ax_corr.tick_params(axis="x", rotation=45)
ax_corr.tick_params(axis="y", rotation=0)
plt.tight_layout()
fig_corr.savefig("matriz_correlacao.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 4. Separar atributos de entrada e variável-alvo ──────────────────────────
X = df1.drop("classe", axis=1)
y = df1["classe"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ── Gráfico 3: Boxplot — Valor do Crédito por Classe ─────────────────────────
fig_box, ax_box = plt.subplots(figsize=(7, 5))
data_box = [df1[df1["classe"] == 0]["valor_credito"],
            df1[df1["classe"] == 1]["valor_credito"]]
bp = ax_box.boxplot(data_box, tick_labels=["Bom pagador (0)", "Mau pagador (1)"],
                    patch_artist=True, notch=False)
for patch, cor in zip(bp["boxes"], ["steelblue", "orange"]):
    patch.set_facecolor(cor)
    patch.set_alpha(0.7)
ax_box.set_title("Valor do Crédito por Classe", fontsize=13, fontweight="bold")
ax_box.set_ylabel("Valor (DM)")
ax_box.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
fig_box.savefig("boxplot_valor_credito.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 5. Normalização ───────────────────────────────────────────────────────────
# Log
X_train_log = X_train.copy(); X_test_log = X_test.copy()
num_cols = ["duracao_meses", "valor_credito", "taxa_parcelamento", "residencia_atual", "idade", "creditos_banco", "dependentes"]
for c in num_cols:
    X_train_log[c] = np.log1p(X_train_log[c])
    X_test_log[c]  = np.log1p(X_test_log[c])

# Min-Max
scaler_mm = MinMaxScaler()
X_train_mm = pd.DataFrame(scaler_mm.fit_transform(X_train), columns=X_train.columns)
X_test_mm  = pd.DataFrame(scaler_mm.transform(X_test),      columns=X_test.columns)

# Z-score
scaler_zs = StandardScaler()
X_train_zs = pd.DataFrame(scaler_zs.fit_transform(X_train), columns=X_train.columns)
X_test_zs  = pd.DataFrame(scaler_zs.transform(X_test),      columns=X_test.columns)

# ── 6-7. Treinar KNN e avaliar influência de k ────────────────────────────────
k_valores = range(1, 31)
accs_mm, accs_zs, accs_log = [], [], [] # acurácia
f1s_mm, f1s_zs, f1s_log = [], [], [] # F1-score
recs_mm, recs_zs, recs_log = [], [], [] # recall

for k in k_valores:
    # Min-Max
    pred_mm = KNeighborsClassifier(k).fit(X_train_mm, y_train).predict(X_test_mm)
    accs_mm.append(accuracy_score(y_test, pred_mm))
    f1s_mm.append(f1_score(y_test, pred_mm))
    recs_mm.append(recall_score(y_test, pred_mm))
 
    # Z-score
    pred_zs = KNeighborsClassifier(k).fit(X_train_zs, y_train).predict(X_test_zs)
    accs_zs.append(accuracy_score(y_test, pred_zs))
    f1s_zs.append(f1_score(y_test, pred_zs))
    recs_zs.append(recall_score(y_test, pred_zs))
 
    # Log
    pred_log = KNeighborsClassifier(k).fit(X_train_log, y_train).predict(X_test_log)
    accs_log.append(accuracy_score(y_test, pred_log))
    f1s_log.append(f1_score(y_test, pred_log))
    recs_log.append(recall_score(y_test, pred_log))

# ── 8-9. Gráfico Ponto de destaque no melhor k  ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("k vs Métricas por Normalização", fontsize=14, fontweight="bold")

metricas_plot = [
    ("Acurácia", accs_mm, accs_zs, accs_log),
    ("F1-score", f1s_mm,  f1s_zs,  f1s_log),
    ("Recall",   recs_mm, recs_zs, recs_log),
]

dados_norm = {
    "Min-Max": {"color": "steelblue", "accs": accs_mm, "f1s": f1s_mm, "recs": recs_mm},
    "Z-score": {"color": "green",     "accs": accs_zs, "f1s": f1s_zs, "recs": recs_zs},
    "Log"    : {"color": "orange",    "accs": accs_log,"f1s": f1s_log, "recs": recs_log},
}

for ax, (titulo, mm, zs, log) in zip(axes, metricas_plot):
    chave = {"Acurácia": "accs", "F1-score": "f1s", "Recall": "recs"}[titulo]
    for nome, d in dados_norm.items():
        valores = d[chave]
        ax.plot(k_valores, valores, label=nome, color=d["color"], linewidth=2)
        best_k  = list(k_valores)[np.argmax(valores)]
        best_v  = max(valores)
        ax.axvline(best_k, color=d["color"], linestyle=":", alpha=0.5)
        ax.scatter([best_k], [best_v], color=d["color"], s=80, zorder=5, edgecolors="white", linewidth=1.5)
        ax.annotate(f"k={best_k}\n{best_v:.3f}",
                    xy=(best_k, best_v),
                    xytext=(best_k + 1, best_v - 0.03),
                    fontsize=7, color=d["color"], fontweight="bold")
    ax.set_title(titulo)
    ax.set_xlabel("k")
    ax.set_ylabel(titulo)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig("k_vs_metricas.png", dpi=150, bbox_inches="tight")
plt.show()
 
# Gráfico 02 

fig_ov, axes_ov = plt.subplots(1, 3, figsize=(16, 5))
fig_ov.suptitle("Acurácia Treino vs Teste por k — Análise de Overfitting",
                fontsize=13, fontweight="bold")

for ax, (nome, Xtr, Xte, accs_te) in zip(axes_ov, [
    ("Min-Max", X_train_mm,  X_test_mm,  accs_mm),
    ("Z-score", X_train_zs,  X_test_zs,  accs_zs),
    ("Log",     X_train_log, X_test_log, accs_log),
]):
    accs_tr = [accuracy_score(y_train, KNeighborsClassifier(k).fit(Xtr, y_train).predict(Xtr))
               for k in k_valores]

    best_k = list(k_valores)[np.argmax(accs_te)]
    best_v = max(accs_te)

    ax.plot(k_valores, accs_tr, label="Treino", color="gray",     linewidth=2, linestyle="--")
    ax.plot(k_valores, accs_te, label="Teste",  color="steelblue",linewidth=2)
    ax.axvline(best_k, color="red", linestyle=":", alpha=0.6)
    ax.scatter([best_k], [best_v], color="red", s=80, zorder=5, edgecolors="white", linewidth=1.5)
    ax.annotate(f"Melhor k={best_k}\n{best_v:.3f}",
                xy=(best_k, best_v), xytext=(best_k + 1, best_v - 0.04),
                fontsize=8, color="red", fontweight="bold")
    ax.set_title(f"{nome}")
    ax.set_xlabel("k")
    ax.set_ylabel("Acurácia")
    ax.set_ylim(0.55, 1.05)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
fig_ov.savefig("overfitting_treino_teste.png", dpi=150, bbox_inches="tight")
plt.show()


# Melhor k por métrica e normalização
print("\n=== MELHOR K (ANÁLISE VISUAL) ===")
for nome, accs, f1s, recs in [("Min-Max", accs_mm, f1s_mm, recs_mm),
                                ("Z-score", accs_zs, f1s_zs, recs_zs),
                                ("Log",     accs_log, f1s_log, recs_log)]:
    print(f"\n{nome}:")
    print(f"  Acurácia → k={list(k_valores)[np.argmax(accs)]}  ({max(accs):.4f})")
    print(f"  F1-score → k={list(k_valores)[np.argmax(f1s)]}   ({max(f1s):.4f})")
    print(f"  Recall   → k={list(k_valores)[np.argmax(recs)]}  ({max(recs):.4f})")


    # ── 10. Validação cruzada --------------------------------------------------------------
print("\n=== VALIDAÇÃO CRUZADA (5-fold) ===")
for nome, Xtr, Xte, f1s in [("Min-Max", X_train_mm,  X_test_mm,  f1s_mm),
                              ("Z-score", X_train_zs,  X_test_zs,  f1s_zs),
                              ("Log",     X_train_log, X_test_log, f1s_log)]:
    melhor_k = list(k_valores)[np.argmax(f1s)]
    knn = KNeighborsClassifier(melhor_k)
    acc_cv = cross_val_score(knn, Xtr, y_train, cv=5, scoring="accuracy")
    f1_cv  = cross_val_score(knn, Xtr, y_train, cv=5, scoring="f1")
    rec_cv = cross_val_score(knn, Xtr, y_train, cv=5, scoring="recall")
    print(f"\n{nome} (k={melhor_k}):")
    print(f"  Acurácia: {acc_cv.mean():.4f} ± {acc_cv.std():.4f}")
    print(f"  F1-score: {f1_cv.mean():.4f}  ± {f1_cv.std():.4f}")
    print(f"  Recall  : {rec_cv.mean():.4f}  ± {rec_cv.std():.4f}")


# ── 11. Grid Search --------------------------------------------------------------------
param_grid = {"n_neighbors": list(range(1, 21)), 
              "weights": ["uniform", "distance"], 
              "metric": ["euclidean", "manhattan"]}

print("\n=== GRID SEARCH ===")
gs_resultados = {}
for nome, Xtr, Xte in [("Min-Max", X_train_mm, X_test_mm),
                        ("Z-score", X_train_zs, X_test_zs),
                        ("Log",     X_train_log, X_test_log)]:
    gs = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="f1")
    gs.fit(Xtr, y_train)
    pred = gs.best_estimator_.predict(Xte)
    gs_resultados[nome] = {"pred": pred, "params": gs.best_params_}
    print(f"\n{nome}:")
    print(f"  Melhores parâmetros: {gs.best_params_}")
    print(f"  Acurácia : {accuracy_score(y_test, pred):.4f}")
    print(f"  F1-score : {f1_score(y_test, pred):.4f}")
    print(f"  Recall   : {recall_score(y_test, pred):.4f}")
    print(f"\n  Classification Report ({nome}):")
    print(classification_report(y_test, pred, target_names=["Bom pagador", "Mau pagador"]))


# ── Gráfico 4: Matrizes de Confusão (uma por normalização) ───────────────────
fig_cm, axes_cm = plt.subplots(1, 3, figsize=(15, 4))
fig_cm.suptitle("Matrizes de Confusão — Grid Search (melhor modelo por normalização)",
                fontsize=13, fontweight="bold")
 
for ax, nome in zip(axes_cm, ["Min-Max", "Z-score", "Log"]):
    cm = confusion_matrix(y_test, gs_resultados[nome]["pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Bom pagador", "Mau pagador"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{nome}\n{gs_resultados[nome]['params']}", fontsize=9)
 
plt.tight_layout()
fig_cm.savefig("matrizes_confusao.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Gráfico 5: Comparação de Acurácia, F1 e Recall entre Normalizações ───────
nomes   = ["Min-Max", "Z-score", "Log"]
accs_gs = [accuracy_score(y_test, gs_resultados[n]["pred"]) for n in nomes]
f1s_gs  = [f1_score(y_test,       gs_resultados[n]["pred"]) for n in nomes]
recs_gs = [recall_score(y_test,   gs_resultados[n]["pred"]) for n in nomes]
 
x  = np.arange(len(nomes))
w  = 0.25
 
fig_comp, ax_comp = plt.subplots(figsize=(9, 5))
b1 = ax_comp.bar(x - w,  accs_gs, w, label="Acurácia",  color="steelblue", alpha=0.85, edgecolor="white")
b2 = ax_comp.bar(x,      f1s_gs,  w, label="F1-score",  color="orange",    alpha=0.85, edgecolor="white")
b3 = ax_comp.bar(x + w,  recs_gs, w, label="Recall",    color="green",     alpha=0.85, edgecolor="white")
 
ax_comp.bar_label(b1, fmt="%.3f", fontsize=8, padding=2)
ax_comp.bar_label(b2, fmt="%.3f", fontsize=8, padding=2)
ax_comp.bar_label(b3, fmt="%.3f", fontsize=8, padding=2)
 
ax_comp.set_xticks(x)
ax_comp.set_xticklabels(nomes)
ax_comp.set_ylim(0, 1.0)
ax_comp.set_ylabel("Valor da Métrica")
ax_comp.set_title("Comparação de Métricas por Normalização (Grid Search)",
                  fontsize=13, fontweight="bold")
ax_comp.legend()
ax_comp.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
fig_comp.savefig("comparacao_normalizacoes.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 12-13. Comparação e análise -------------------------------------------------------------------------------
print("\n=== RESUMO FINAL ===")
print(f"{'Normalização':<12} {'Acurácia':>10} {'F1-score':>10} {'Recall':>10}")
print("-" * 44)
for nome, accs, f1s, recs in [("Min-Max", accs_mm, f1s_mm, recs_mm),
                                ("Z-score", accs_zs, f1s_zs, recs_zs),
                                ("Log",     accs_log, f1s_log, recs_log)]:
    print(f"  {nome:<10} {max(accs):>10.4f} {max(f1s):>10.4f} {max(recs):>10.4f}")

