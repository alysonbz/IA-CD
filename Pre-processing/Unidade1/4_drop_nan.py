## realize print do dataset volunteer corrigido sem nenhum NAN, para isto removam as colunas NAN e depois as linhas e crie
# um dataframe novo e print este mostrando a contagem de colunas NAN existentes e mostre também o shape novo.

from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

print(volunteer.info)

volunteer = volunteer.dropna(axis=1, how='all')
print(volunteer)

volunteer = volunteer.dropna()
print(volunteer)

print(volunteer.isna().sum())

print(volunteer.shape)

print(volunteer.info)
