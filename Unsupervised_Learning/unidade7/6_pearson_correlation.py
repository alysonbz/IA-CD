# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from src.utils import load_grains_dataset

# Carrega o dataset
grains_df = load_grains_dataset()

# Atribui as colunas de interesse
width = grains_df.iloc[:, 0]
length = grains_df.iloc[:, 1]

# Gráfico de dispersão entre largura e comprimento
plt.scatter(width, length)
plt.xlabel("Grain width")
plt.ylabel("Grain length")
plt.axis('equal')
plt.title("Scatter plot - Width vs Length")
plt.show()

# Calcula a correlação de Pearson
correlation, pvalue = pearsonr(width, length)

# Exibe o valor da correlação
print("Correlação de Pearson:", round(correlation, 3))