# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from src.utils import load_grains_dataset


# Carregar os dados
grains_df = load_grains_dataset()

# Atribuir a 0ª coluna como largura e a 1ª como comprimento
width = grains_df['0']
length = grains_df['1']

# Gráfico de dispersão entre largura e comprimento
plt.scatter(width, length)
plt.axis('equal')

plt.show()

# Calcular a correlação de Pearson
correlation, pvalue = pearsonr(width, length)

# Exibir o valor da correlação
print(correlation)
