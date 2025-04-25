from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Print os primeiros elementos da coluna hits
print(volunteer['hits'].head())

# Print as características da coluna hits
print(volunteer['hits'].info(), '\n')

# Converta a coluna hits para o tipo int e salve no DataFrame
volunteer['hits'] = volunteer['hits'].astype('int64')

# Print as características da coluna hits novamente
print(volunteer['hits'].info())

