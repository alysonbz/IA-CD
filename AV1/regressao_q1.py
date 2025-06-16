# 1. Carregue o dataset.

import pandas as pd 

df2 = pd.read_csv("/content/Concrete_Data_Yeh.csv")
df2

##

# 2. Trate valores ausentes
print("Valores Ausentes:", df2.isnull().sum())

##

# 3. Verifique a correlação com a variável-alvo.
correlacao = df2.corr()["csMPa"].sort_values(ascending=False)
print(correlacao)

##

# 4. Normalize ou padronize os dados.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df2_scaled = pd.DataFrame(scaler.fit_transform(df2), columns=df2.columns)
df2_scaled  

