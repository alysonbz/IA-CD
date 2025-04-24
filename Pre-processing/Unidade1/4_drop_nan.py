from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()
print(volunteer)

## realize print do dataset volunteer corrigido sem nenhum NAN, para isto removam as colunas NAN e depois as linhas e crie

volunteerNAN = volunteer.dropna(axis=1)
print(volunteerNAN)

#um dataframe novo e print este mostrando a contagem de colunas NAN existentes e mostre também o shape novo.

volunteerNAN2 = volunteerNAN.dropna(axis=0)
print(volunteerNAN2)

print(volunteerNAN2.isna().sum())
print(volunteerNAN2.shape)
print(volunteerNAN2.head())