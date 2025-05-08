from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Print os primeiros elementos da coluna hits
print(volunteer["hits"].head())
print("")

# Print as caracteristicas da coluna hits
print(volunteer["hits"].describe())
print("")

# Converta a coluna hits para o tipo int
volunteer["hits"] = volunteer["hits"].astype("int64")
print(volunteer["hits"].dtypes)
print("")

# Print as caracteristicas da coluna hits novamente
print(volunteer["hits"].describe)
print("")
