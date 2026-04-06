from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# 1. Print os primeiros elementos da coluna hits
print("\nQuestão 1.")
print(volunteer['hits'].head())

# 2. Print as caracteristicas da coluna hits
print("\nQuestão 2.")
print(volunteer['hits'].describe())
print(volunteer['hits'].dtype)

# 3. Converta a coluna hits para o tipo int32
volunteer['hits'] = volunteer['hits'].astype('int32')

# 4. Print as caracteristicas da coluna hits novamente
print("\nQuestão 4.")
print(volunteer['hits'].describe())
print(volunteer['hits'].dtype)