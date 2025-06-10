from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

print('shape antigo: ',volunteer.shape)
volunteer_nan = volunteer.isna().sum()
volunteer_nan_count = []
for i in volunteer_nan:
    if i > 0:
        volunteer_nan_count.append(i)

print('número de colunas com valores NaN existentes: ',len(volunteer_nan_count))
# retirando colunas vazias
volunteer = volunteer.dropna(axis=1)

# retirando linhas vazias
volunteer = volunteer.dropna()

## realize print do dataset volunteer corrigido sem nenhum NAN, para isto removam as colunas NAN e depois as linhas e crie
#um dataframe novo e print este mostrando a contagem de colunas NAN existentes e mostre também o shape novo.

print('shape novo: ',volunteer.shape)
