from src.utils import load_hiking_dataset , load_df2_unidade1,load_wine_dataset, load_df1_unidade1, load_volunteer_dataset
import pandas as pd

volunteer = load_volunteer_dataset()
hiking = load_hiking_dataset()
wine  = load_wine_dataset()
df1 = load_df1_unidade1()
df2 = load_df2_unidade1()


print(wine.describe())
print(wine.info())
print(df1)
print("\n",df1.dropna())
print("\n",df1.drop([1,2,4]))
print("\n",df1.isna().sum())
print("\n",df1.dropna(subset=["B"]))
print("\n",df1.dropna(thresh=2))


print("\n\nShape: ",volunteer.shape)

print("\n",volunteer.info())

lista=[]
with open('../dataset/wine.csv', 'r') as f:
    for linha in f.readlines()[1:]:
        a = linha.strip().split(',')
        for i in range(4):
            a[i] = float(a[i])

        lista.append(a)
print(lista)
def countclasses(lista):
    c1 = 0
    c2 = 0
    c3 = 0
    for i in range(len(lista)):
        if lista[i][0] == 1.0:
            c1 += 1
        if lista[i][0] == 2.0:
            c2 += 1
        if lista[i][0] == 3.0:
            c3 += 1
    return [c1, c2, c3]
print(countclasses(lista))

p=0.6
c1,c2, c3 = countclasses(lista)
treinamento, teste= [], []
max_c1, max_c2, max_c3 = int(p*c1), int(p*c2), int(p*c3)
total1 =0
total2 =0
total3 =0
for lis in lista:
    if lis[0]==1.0 and total1< max_c1:
        treinamento.append(lis)
        total1 +=1
    elif lis[0]==2.0 and total2<max_c2:
        treinamento.append(lis)
        total2 +=1
    elif lis[0]==3.0 and total3<max_c3:
        treinamento.append(lis)
        total3 +=1
    else:
        teste.append(lis)
print(total1, total2, total3)
print(len(treinamento),len(teste), len(lista))

import math
def dist_euclidiana(v1,v2):
    dim, soma = len(v1), 0
    for i in range(dim -1):
        soma += math.pow(v1[i] -v2[i],2)
    return math.sqrt(soma)