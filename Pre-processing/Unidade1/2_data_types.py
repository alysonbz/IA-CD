from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Print os primeiros elementos da coluna hits
print(volunteer["hits"].drop(5))

# Print as caracteristicas da coluna hits
print(volunteer)

# Converta a coluna hits para o tipo int
print(volunteer)

# Print as caracteristicas da coluna hits novamente
print()
