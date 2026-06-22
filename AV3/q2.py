import os

import matplotlib.pyplot as plt
import seaborn as sns

from q1 import load_data, preprocess_data


def plot_pairplot(df):
    g = sns.pairplot(df)
    g.savefig("image/q2/pairplot_geral.png")
    plt.close(g.fig)


def plot_kde_pairs(df):
    pairs = [  # Melhores candidatos que encontrei
        ("PRC_FULL_PAYMENT", "CREDIT_LIMIT"),
        ("CASH_ADVANCE", "PURCHASES_FREQUENCY"),
        ("BALANCE", "PURCHASES"),
        ("BALANCE", "PRC_FULL_PAYMENT"),
        ("MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT"),
        ("PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY"),
        ("ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES"),
        ("PURCHASES_TRX", "CASH_ADVANCE_TRX"),
        ("CREDIT_LIMIT", "BALANCE"),
        ("CREDIT_LIMIT", "PAYMENTS"),
        ("CASH_ADVANCE", "PRC_FULL_PAYMENT"),
        ("BALANCE", "MINIMUM_PAYMENTS"),
    ]

    n_cols = 3
    rows = len(pairs) // n_cols + (1 if len(pairs) % n_cols else 0)

    fig, axes = plt.subplots(rows, n_cols, figsize=(18, rows * 5))
    axes = axes.flatten()

    for i, (x_col, y_col) in enumerate(pairs):
        sns.kdeplot(data=df, x=x_col, y=y_col, cmap="viridis", fill=True, ax=axes[i])
        axes[i].set_title(f"{x_col}  ×  {y_col}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel(x_col, fontsize=9)
        axes[i].set_ylabel(y_col, fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("KDE - Combinações de Atributos", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("image/q2/kde_pares_atributos.png")
    plt.close(fig)


def plot_scatter_two_attributes(df):
    fig = plt.figure()
    df.plot(kind="scatter", x="PRC_FULL_PAYMENT", y="CREDIT_LIMIT", alpha=0.5, color="steelblue")
    plt.savefig("image/q2/scatter_prc_full_payment_credit_limit.png")
    plt.close(fig)


def pipeline():
    os.makedirs("image/q2", exist_ok=True)

    df = preprocess_data(load_data())

    plot_pairplot(df)
    plot_kde_pairs(df)
    plot_scatter_two_attributes(df)

    return df


if __name__ == "__main__":
    pipeline()
