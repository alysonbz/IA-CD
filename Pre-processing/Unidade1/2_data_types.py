from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Print os primeiros elementos da coluna hits
print(f'-Essas são as primeiras linhas da coluna hits-\n{volunteer['hits'].head()}\n')

# Print as caracteristicas da coluna hits
print(f'-Essa são as características da coluna hits-'
      f'\nTipo de coluna: {volunteer['hits'].dtype}'
      f'\nValores faltando: {volunteer['hits'].isna().sum()}'
      f'\nResumo geral: {volunteer['hits'].describe()}')


# Converta a coluna hits para o tipo int
volunteer['hits'] = volunteer['hits'].astype('int32')

# Print as caracteristicas da coluna hits novamente
print(f'-Características atualizadas da coluna hits-'
      f'\nTipo de coluna: {volunteer['hits'].dtype}'
      f'\nValores faltando: {volunteer['hits'].isna().sum()}'
      f'\nResumo geral: {volunteer['hits'].describe()}')
