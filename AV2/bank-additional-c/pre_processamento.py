from sklearn.preprocessing import LabelEncoder


def preprocessar_dados(df):

    # Removendo unknown

    for coluna in df.columns:
        df = df[df[coluna] != 'unknown']

    print('Shape após remoção de unknown:', df.shape)

    # Resetando índice
    df = df.reset_index(drop=True)

    # Convertendo variáveis categóricas

    encoder = LabelEncoder()

    colunas_categoricas = df.select_dtypes(include=['object']).columns

    print('Colunas categóricas encontradas:')
    print(colunas_categoricas)

    for coluna in colunas_categoricas:

        df[coluna] = encoder.fit_transform(df[coluna])

        print(f'Coluna {coluna} convertida!')

    print('Pré-processamento concluído!')

    return df