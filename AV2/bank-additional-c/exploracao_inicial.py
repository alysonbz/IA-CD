import matplotlib.pyplot as plt
import seaborn as sns


def explorar_dados(df):

    print('Head do Dataset:')
    print(df.head())

    print('Informações:')
    print(df.info())

    print('Descrição:')
    print(df.describe())

    print('Valores nulos:')
    print(df.isnull().sum())

    print('Formato do Dataset:')
    print(df.shape)


def mapa_correlacao(df):

    plt.figure(figsize=(16,10))

    sns.heatmap(
        df.corr(),
        annot=True,
        fmt='.2f',
        cmap='coolwarm'
    )

    plt.title('Mapa de Correlação')
    plt.show()