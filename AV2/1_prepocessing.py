import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Carregando o dataset
df = pd.read_csv('dataset/marketing_campaign.csv', sep='\t')

# Remover nulos
df = df.dropna(subset=['Income'])

# Renomear colunas
df = df.rename(columns={
    "MntWines": "Wines",
    "MntFruits": "Fruits",
    "MntMeatProducts": "Meat",
    "MntFishProducts": "Fish",
    "MntSweetProducts": "Sweets",
    "MntGoldProds": "Gold"
})

# Converter data
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)
most_recent = df['Dt_Customer'].max()
df['Customer_For'] = (most_recent - df['Dt_Customer']).dt.days // 365

# Agrupar escolaridade
df["Education"] = df["Education"].replace({
    "Basic": "Undergraduate",
    "2n Cycle": "Undergraduate",
    "Graduation": "Graduate",
    "Master": "Postgraduate",
    "PhD": "Postgraduate"
})

# Agrupar estado civil
df["Living_With"] = df["Marital_Status"].replace({
    "Married": "Partner",
    "Together": "Partner",
    "Absurd": "Alone",
    "Widow": "Alone",
    "YOLO": "Alone",
    "Divorced": "Alone",
    "Single": "Alone"
})

# Novos atributos
df['Age'] = 2025 - df['Year_Birth']
df["Spent"] = df["Wines"] + df["Fruits"] + df["Meat"] + df["Fish"] + df["Sweets"] + df["Gold"]
df["Children"] = df["Kidhome"] + df["Teenhome"]
df["Family_Size"] = df["Living_With"].replace({"Alone": 1, "Partner": 2}) + df["Children"]
df["Is_Parent"] = np.where(df.Children > 0, 1, 0)

# Remover colunas irrelevantes
to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
df = df.drop(columns=to_drop)

# Codificar variáveis categóricas
label_cols = df.select_dtypes(include='object').columns
for col in label_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Remover outliers
df = df[df['Age'] < 90]
df = df[df['Income'] < 600000]

# Salvar
df.to_csv('dataset/marketing_campaign_preprocessed.csv', index=False)
print("Pré-processamento concluído.")

# Histogramas
df.hist(bins=30, figsize=(20, 15), color='steelblue', edgecolor='black')
plt.suptitle("Distribuições das variáveis numéricas", fontsize=20)
plt.tight_layout()
plt.show()

# Boxplots para variáveis principais
import seaborn as sns
sns.set(style="whitegrid")
To_Plot = ["Income", "Recency", "Customer_For", "Age", "Spent", "Is_Parent"]

plt.figure(figsize=(20, 8))
for i, col in enumerate(To_Plot, 1):
    plt.subplot(1, len(To_Plot), i)
    sns.boxplot(y=df[col], color="lightblue")
    plt.title(col)
plt.tight_layout()
plt.show()

# Heatmap de correlação
import numpy as np
plt.figure(figsize=(10, 8))
correlation = df.corr(numeric_only=True)
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Calor das Correlações')
plt.show()

