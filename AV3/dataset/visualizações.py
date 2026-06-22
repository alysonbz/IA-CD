from dataset.pre_processamento import *
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

n_dados = [len(batch1), len(batch2), len(batch3), len(batch4), len(batch5), len(batch6), len(batch7), len(batch8), len(batch9), len(batch10)]
nome_dados = ['batch1','batch2','batch3','batch4','batch5','batch6','batch7','batch8','batch9','batch10']
n_gas = [
    len(all_batch[all_batch['gas']=='Etanol']),
    len(all_batch[all_batch['gas']=='Etileno']),
    len(all_batch[all_batch['gas']=='Amônia']),
    len(all_batch[all_batch['gas']=='Acetaldeído']),
    len(all_batch[all_batch['gas']=='Acetona']),
    len(all_batch[all_batch['gas']=='Metilbenzeno'])
]
nome_gas = ['Etanol','Etileno','Amônia','Acetaldeído','Acetona','Metilbenzeno']

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].bar(nome_dados,n_dados, color='#8494B8')
ax[0].tick_params(rotation=45)
ax[0].set_title('Amostras por Batch')

ax[1].bar(nome_gas,n_gas, color=['#BA3838','#59339E','#596BA8','#3D828A','#327346','#AB7620'])
ax[1].tick_params(rotation=45)
ax[1].set_title('Amostras por Gás')

plt.show()