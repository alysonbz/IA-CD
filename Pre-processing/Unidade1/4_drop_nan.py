from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

volunteer_cleaned = volunteer.dropna(axis=1)
volunteer_cleaned = volunteer_cleaned.dropna(axis=0)
print(volunteer_cleaned.isnull().sum())
print(volunteer_cleaned.shape)

## realize print do dataset volunteer corrigido sem nenhum NAN, para isto removam as colunas NAN e depois as linhas e crie
#um dataframe novo e print este mostrando a contagem de colunas NAN existentes e mostre também o shape novo.