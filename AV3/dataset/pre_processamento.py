import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

batch1 = pd.read_csv('../dataset/batch1.dat', sep=' ', header=None)
batch2 = pd.read_csv('../dataset/batch2.dat', sep=' ', header=None)
batch3 = pd.read_csv('../dataset/batch3.dat', sep=' ', header=None)
batch4 = pd.read_csv('../dataset/batch4.dat', sep=' ', header=None)
batch5 = pd.read_csv('../dataset/batch5.dat', sep=' ', header=None)
batch6 = pd.read_csv('../dataset/batch6.dat', sep=' ', header=None)
batch7 = pd.read_csv('../dataset/batch7.dat', sep=' ', header=None)
batch8 = pd.read_csv('../dataset/batch8.dat', sep=' ', header=None)
batch9 = pd.read_csv('../dataset/batch9.dat', sep=' ', header=None)
batch10 = pd.read_csv('../dataset/batch10.dat', sep=' ', header=None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 20)

def transformar_data(linhas):
    gas, concentracao = linhas[0].split(';')
    concentracao = float(concentracao)
    gas = int(gas)
    features = {}

    for i in range(1, len(linhas)):
        if pd.notna(linhas[i]):
            index, valor = linhas[i].split(':')
            features[f"feature_{int(index)}"] = float(valor)

    return {"gas": gas, "concentracao": concentracao, **features}
batch1 = batch1.apply(transformar_data, axis=1, result_type='expand')
batch2 = batch2.apply(transformar_data, axis=1, result_type='expand')
batch3 = batch3.apply(transformar_data, axis=1, result_type='expand')
batch4 = batch4.apply(transformar_data, axis=1, result_type='expand')
batch5 = batch5.apply(transformar_data, axis=1, result_type='expand')
batch6 = batch6.apply(transformar_data, axis=1, result_type='expand')
batch7 = batch7.apply(transformar_data, axis=1, result_type='expand')
batch8 = batch8.apply(transformar_data, axis=1, result_type='expand')
batch9 = batch9.apply(transformar_data, axis=1, result_type='expand')
batch10 = batch10.apply(transformar_data, axis=1, result_type='expand')
all_batch = pd.concat([batch1, batch2, batch3, batch4,batch5,batch6,batch7,batch8,batch9,batch10], ignore_index=True)
#print(all_batch['gas'].value_counts().sort_index())
all_batch_numbers = all_batch.copy()
all_batch['gas'] = all_batch['gas'].replace({
    1.0: 'Etanol',
    2.0: 'Etileno',
    3.0: 'Amônia',
    4.0: 'Acetaldeído',
    5.0: 'Acetona',
    6.0: 'Metilbenzeno'
})

X = all_batch[[f'feature_{i}' for i in range(1,129)]]
y = all_batch['gas']
F, p = f_classif(X, y)
ranking = pd.DataFrame({
    'feature': X.columns,
    'F': F
}).sort_values('F', ascending=False)

scaler = StandardScaler()
minmax_scaler = MinMaxScaler(feature_range=(0,1))

selecionadas = []
limite = 0.9
for feat in ranking['feature']:
    if len(selecionadas) == 0:
        selecionadas.append(feat)
    else:
        correlacoes = [
            abs(all_batch[[feat, f]].corr().iloc[0,1])
            for f in selecionadas
        ]
        if max(correlacoes) < limite:
            selecionadas.append(feat)
    if len(selecionadas) == 20:
        break

X = all_batch[selecionadas].values
X_ss = scaler.fit_transform(X)
X_mm = minmax_scaler.fit_transform(X)
y = all_batch_numbers[['gas']]