#!pip install kagglehub
#!pip install yellowbrick

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import kagglehub
from kagglehub import KaggleDatasetAdapter


def load_data():
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "arjunbhasin2013/ccdata",
        "CC GENERAL.csv",
    )
    return df


def check_duplicates(df):
    print(df.duplicated().sum())


def check_negative_values(df):
    # checando por valores negativos
    print((df.select_dtypes(include=np.number) < 0).sum())


def check_frequency_bounds(df):
    # checando por valores negativos em colunas de frequência
    freq_cols = [
        "BALANCE_FREQUENCY",
        "PURCHASES_FREQUENCY",
        "ONEOFF_PURCHASES_FREQUENCY",
        "PURCHASES_INSTALLMENTS_FREQUENCY",
        "CASH_ADVANCE_FREQUENCY",
        "PRC_FULL_PAYMENT",
    ]

    print(((df[freq_cols] < 0) | (df[freq_cols] > 1)).sum())
    return freq_cols


def show_invalid_cash_advance_rows(df):
    print(df[df['CASH_ADVANCE_FREQUENCY'] > 1])


def plot_continuous_distributions(df):
    continuous_cols = [
        "BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES",
        "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "PURCHASES_FREQUENCY",
        "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY",
        "CASH_ADVANCE_FREQUENCY", "CREDIT_LIMIT", "PAYMENTS",
        "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT",
    ]

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()

    for ax, col in zip(axes, continuous_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue")
        ax.set_title(col)

    for j in range(len(continuous_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Distribuição das Variáveis Contínuas", y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig("image/q1/distribuicao_variaveis_continuas.png")
    plt.close(fig)


def drop_cust_id(df):
    df = df.drop(columns=["CUST_ID"])
    return df


def describe_data(df):
    print(df.describe().T)


def show_info(df):
    df.info()


def get_missing_value_summary(df):
    missing_values_ = df.isnull().sum().sort_values(ascending=False).head()
    columns_with_missing_values = missing_values_[missing_values_ > 0].index
    return missing_values_, columns_with_missing_values


def print_missing_counts(missing_values_, columns_with_missing_values):
    # numero de valores faltantes por coluna
    print(missing_values_[columns_with_missing_values])


def print_missing_percentage(df):
    # porcentagem de valores faltantes
    print((df.isnull().sum().sort_values(ascending=False).head() / df.shape[0]) * 100)


def show_credit_limit_nulls(df):
    print(df[df['CREDIT_LIMIT'].isnull()])


def print_credit_limit_median(df):
    print(df['CREDIT_LIMIT'].median())


def drop_credit_limit_nulls(df):
    # removido pois a maioria dos valores faltantes estão concentrados nessa coluna e a mediana pode não ser representativa
    df = df.dropna(subset=['CREDIT_LIMIT'])
    return df


def impute_missing_values(df, columns_with_missing_values):
    imputer = SimpleImputer(strategy="median")
    pipeline = Pipeline(steps=[("imputer", imputer)])

    df[columns_with_missing_values] = pipeline.fit_transform(df[columns_with_missing_values])
    return df


def plot_discrete_distributions(df):
    count_cols = ['CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'TENURE']

    fig, axes = plt.subplots(1, len(count_cols), figsize=(16, 5))

    for ax, col in zip(axes, count_cols):
        df[col].value_counts().sort_values(ascending=False).head(30).plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
        ax.set_title(col)
        ax.set_xlabel("Valor")
        ax.set_ylabel("Frequência")
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("Distribuição das Variáveis Discretas", y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig("image/q1/distribuicao_variaveis_discretas.png")
    plt.close(fig)


def plot_boxplot(df):
    fig = plt.figure()
    sns.boxplot(data=df)
    plt.xticks(rotation=45, ha="right")
    plt.savefig("image/q1/boxplot_variaveis.png")
    plt.close(fig)


def print_tenure_analysis(df):
    print(f"TENURE - valores únicos: {sorted(df['TENURE'].unique())}")
    print(f"TENURE - desvio padrão: {df['TENURE'].std():.3f}")

    print((df["TENURE"].value_counts(normalize=True).sort_index() * 100).round(2))


def preprocess_data(df):
    df = drop_cust_id(df)
    _, columns_with_missing_values = get_missing_value_summary(df)
    df = drop_credit_limit_nulls(df)
    df = impute_missing_values(df, columns_with_missing_values)
    return df


def pipeline():
    os.makedirs("image/q1", exist_ok=True)

    df = load_data()
    print(df.head())

    check_duplicates(df)
    check_negative_values(df)
    check_frequency_bounds(df)
    show_invalid_cash_advance_rows(df)
    plot_continuous_distributions(df)

    df = drop_cust_id(df)

    describe_data(df)
    show_info(df)

    missing_values_, columns_with_missing_values = get_missing_value_summary(df)
    print_missing_counts(missing_values_, columns_with_missing_values)
    print_missing_percentage(df)

    show_credit_limit_nulls(df)
    print_credit_limit_median(df)

    df = drop_credit_limit_nulls(df)
    df = impute_missing_values(df, columns_with_missing_values)

    plot_discrete_distributions(df)
    plot_boxplot(df)
    print_tenure_analysis(df)

    return df


if __name__ == "__main__":
    pipeline()
