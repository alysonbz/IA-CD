import pandas as pd

def carregar_dataset():
    df = pd.read_csv('bank-additional-full.csv', sep=';')
    print('dados:',df.head())
    return df
