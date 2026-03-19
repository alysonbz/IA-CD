from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Print os primeiros elementos da coluna hits
hits = volunteer['hits']
print(hits.head(5))
# Print as caracteristicas da coluna hits
print(hits.describe())
print(hits.info())


# Converta a coluna hits para o tipo int
hits_int = hits.astype("int32")
print(hits_int.head(5))

# Print as caracteristicas da coluna hits novamente
print(hits_int.describe())
print(hits.info())
