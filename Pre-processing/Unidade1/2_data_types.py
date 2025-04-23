from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Print os primeiros elementos da coluna hits

print(volunteer["hits"].head())

# Print as caracteristicas da coluna hits

print(volunteer.describe())

# Converta a coluna hits para o tipo int

# Converta a coluna hits para o tipo int

volunteer["hits"] = volunteer["hits"].astype("int")


# Print as caracteristicas da coluna hits novamente

print(volunteer.describe())