from src.utils import load_hiking_dataset , load_df2_unidade1,load_wine_dataset, load_df1_unidade1, load_volunteer_dataset
import pandas

volunteer = load_volunteer_dataset()
hiking = load_hiking_dataset()
wine  = load_wine_dataset()
df1 = load_df1_unidade1()
df2 = load_df2_unidade1()


#print(wine.describe())
#print((wine).info())
#print((df1).info())
#print(df1.drop([1, 2, 3]))
#print(df1.drop("A", axis=1))
#print(df1.isna().sum())
#print(df1.dropna(subset=["B"]))
#print(df1.dropna(thresh=2))
#print(df2)
#print(df2.inf())
#df2["C"] = df2["C"].astype("int64")

from sklearn.model_selection import train_test_split

X = volunteer.drop('category_desc', axis=1)
y = volunteer['category_desc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Treino',y_train.value_counts())

print('Teste:',y_test.value_counts())





#ATIVIDADE
lista = []
with open(r'C:\Users\Usuario\Downloads\iris\iris.data', 'r') as f:
    for linha in f.readlines():
        linha = linha.strip()
        if linha == '':
            continue
        a = linha.split(',')
        lista.append(a)


        a[0] = float(a[0])
        a[1] = float(a[1])
        a[2] = float(a[2])
        a[3] = float(a[3])


def countclasses(lista):
    setosa = 0
    versicolor = 0
    virginica = 0
    for i in range(len(lista)):
        if lista[i][4] == 'Iris-setosa':
            setosa += 1
        elif lista[i][4] == 'Iris-versicolor':
            versicolor += 1
        elif lista[i][4] == 'Iris-virginica':
            virginica += 1

    return [setosa, versicolor, virginica]


p=0.6
setosa,versicolor, virginica = countclasses(lista)
treinamento, teste= [], []
max_setosa, max_versicolor, max_virginica = int(p*setosa), int(p*versicolor), int(p*virginica)
total1 =0
total2 =0
total3 =0
for lis in lista:
    if lis[-1] == 'Iris-setosa' and total1 < max_setosa:
        treinamento.append(lis)
        total1 += 1
    elif lis[-1] == 'Iris-versicolor' and total2 < max_versicolor:
        treinamento.append(lis)
        total2 += 1
    elif lis[-1] == 'Iris-virginica' and total3 < max_virginica:
        treinamento.append(lis)
        total3 += 1
    else:
        teste.append(lis)


import math
def dist_euclidiana(v1,v2):
    dim, soma = len(v1), 0
    for i in range(dim -1):
        soma += math.pow(v1[i] -v2[i],2)
    return math.sqrt(soma)


def knn(treinamento, nova_amostra, K):
    dists, len_treino = {}, len(treinamento)

    for i in range(len_treino):
        d = dist_euclidiana(treinamento[i], nova_amostra)
        dists[i] = d

    k_vizinhos = sorted(dists, key=dists.get)[:K]

    qtd_setosa, qtd_versicolor, qtd_virginica = 0, 0, 0

    for indice in k_vizinhos:
        if treinamento[indice][-1] == 'Iris-setosa':
            qtd_setosa += 1
        elif treinamento[indice][-1] == 'Iris-versicolor':
            qtd_versicolor += 1
        elif treinamento[indice][-1] == 'Iris-virginica':
            qtd_virginica += 1

    a = [qtd_setosa, qtd_versicolor, qtd_virginica]

    if a.index(max(a)) == 0:
        return 'Iris-setosa'
    elif a.index(max(a)) == 1:
        return 'Iris-versicolor'
    else:
        return 'Iris-virginica'

acertos, K = 0,1
for amostra in teste:
    classe = knn(treinamento, amostra, K)
    if amostra[-1]==classe:
        acertos +=1
print("Porcentagem de acertos:",100*acertos/len(teste))