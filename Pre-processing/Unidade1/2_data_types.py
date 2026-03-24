from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Print os primeiros elementos da coluna hits
print("\nQ1")
print(volunteer['hits'].head())

# Print as caracteristicas da coluna hits
print("\nQ2")
volunteer[['hits']].info()

# Converta a coluna hits para o tipo int
volunteer['hits'] = volunteer['hits'].astype(int)

# Print as caracteristicas da coluna hits novamente
print("\nQ4")
print(volunteer['hits'].head())
