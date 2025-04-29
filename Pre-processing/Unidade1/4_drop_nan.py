from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

## realize print do dataset volunteer corrigido sem nenhum NAN, para isto removam as colunas NAN e depois as linhas e crie
#um dataframe novo e print este mostrando a contagem de colunas NAN existentes e mostre também o shape novo.
# retirando os nan das colunas
volunteer = volunteer.dropna(axis=1)
#retirando os nan das linhas
volunteer = volunteer.dropna()
#fazendo as contagens de colunas nan
print(volunteer.isna().sum())
#mostrando shape.
print(f'esse é o shape = {volunteer.shape}')
