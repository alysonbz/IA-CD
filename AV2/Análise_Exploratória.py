import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ajustar estilo dos gráficos
sns.set(style="whitegrid")

# Carregar o dataset
file_path = "StudentsPerformance.csv"
df = pd.read_csv(file_path)

# Exibir informações básicas
print("== INFORMAÇÕES DO DATAFRAME ==")
print(df.info(), "\n")

print("== PRIMEIRAS LINHAS DO DATASET ==")
print(df.head(), "\n")

print("== DESCRIÇÃO ESTATÍSTICA COMPLETA ==")
print(df.describe(include='all'), "\n")

# Distribuição das três notas
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, col in zip(axes, ['math score', 'reading score', 'writing score']):
    sns.histplot(df[col], kde=True, bins=20, ax=ax, color="steelblue")
    ax.set_title(f'Distribuição de {col}')
plt.tight_layout()
plt.show()

# Boxplots para verificar outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['math score', 'reading score', 'writing score']])
plt.title('Boxplot das Notas')
plt.show()

# Heatmap de correlação
corr = df[['math score', 'reading score', 'writing score']].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlação entre as Notas')
plt.show()

# Mostrar a matriz de correlação no console
print("== MATRIZ DE CORRELAÇÃO ENTRE AS NOTAS ==")
print(corr)
