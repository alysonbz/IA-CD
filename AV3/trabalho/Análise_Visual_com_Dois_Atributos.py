from dataset.pre_processamento import *
import matplotlib.pyplot as plt
import seaborn as sns

print(ranking.head(20))

sns.heatmap(all_batch[ranking['feature'].head(20)].corr(), cmap='coolwarm')

plt.figure(figsize=(8,6))
gases = ['Etanol','Etileno','Amônia','Acetaldeído','Acetona','Metilbenzeno']
for gas in gases:
    dados_gas = all_batch[all_batch['gas'] == gas]
    plt.scatter(
        dados_gas['feature_9'],#9#67
        dados_gas['feature_100'],#100#11
        label=f'{gas}', alpha=0.7
    )
plt.xlabel('Feature 9')
plt.ylabel('Feature 100')
plt.legend()
plt.show()