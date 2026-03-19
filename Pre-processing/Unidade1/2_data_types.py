from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Print os primeiros elementos da coluna hits
volunteer['hits'].head()

# Print as caracteristicas da coluna hits
volunteer['hits'].describe().T

# Converta a coluna hits para o tipo int
volunteer['hits'] = volunteer['hits'].astype('int32')

# Print as caracteristicas da coluna hits novamente
volunteer['hits'].describe().T
