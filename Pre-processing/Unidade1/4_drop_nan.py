from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

## realize print do dataset volunteer corrigido sem nenhum NAN, para isto removam as colunas NAN e depois as linhas e crie
#um dataframe novo e print este mostrando a contagem de colunas NAN existentes e mostre também o shape novo.
volunteer_sem_linhas = volunteer.dropna(axis=1)
volunteer_limpo = volunteer_sem_linhas.dropna(axis=0)
count_nan = volunteer_limpo.isnull().sum()

print('Contagem da NaNs por coluna no novo Dataframe')
print(count_nan)

print('\nShape do Dataframe Original')
print(volunteer.shape)

print('\nShape do Novo Dataframe sem os NaNs')
print(volunteer_limpo.shape)