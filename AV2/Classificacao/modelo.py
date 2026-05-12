import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', None)
df = pd.read_csv('dataset_tratado.csv')
#print(df.head())

#Separação das colunas alvo e teste
X = df.drop('Revenue', axis=1)
y = df['Revenue']

scaler = StandardScaler()
X_zscore = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_zscore, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print('Acurácia:', accuracy_score(y_test, y_pred))