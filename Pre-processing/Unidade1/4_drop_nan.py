from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# retirando colunas vazias
volunteer = volunteer.dropna(axis=1)

# retirando linhas vazias
volunteer = volunteer.dropna()

## realize print do dataset volunteer corrigido sem nenhum NAN, para isto removam as colunas NAN e depois as linhas e crie
#um dataframe novo e print este mostrando a contagem de colunas NAN existentes e mostre também o shape novo.
print(volunteer.isna().sum())
print(volunteer.shape)
