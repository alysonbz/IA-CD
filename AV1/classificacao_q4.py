import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --------------------
# 1. Ler e preparar os dados
# --------------------
df = pd.read_csv("IA-CD/AV1/bancos/flavors_of_cacao.csv")
df.columns = [
    "Company", "Specific_Bean_Origin", "REF", "Review_Date", "Cocoa_Percent",
    "Company_Location", "Rating", "Bean_Type", "Broad_Bean_Origin"
]
# Limpar e renomear colunas para facilitar
df.columns = [col.strip() for col in df.columns]  # Remove espaços

# Transformar a nota (Rating) em classe: Exemplo - acima de 3.0 é bom, abaixo é ruim
df['target'] = (df['Rating'] >= 3.0).astype(int)

# Selecionar features numéricas ou codificar categóricas
df = df.dropna()  # Remover linhas com valores faltantes
X = df.drop(columns=['Company', 'Specific_Bean_Origin', 'REF', 'Review_Date', 'Cocoa_Percent', 'Company_Location', 'Rating', 'target'])
X = pd.get_dummies(X)  # Codificar variáveis categóricas
y = df['target']

# --------------------
# 2. Dividir em treino e teste
# --------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --------------------
# 3. Criar pipeline com normalização + KNN
# --------------------
pipe = Pipeline([
    ('scaler', StandardScaler()),  # Placeholder, o GridSearch vai testar os outros
    ('knn', KNeighborsClassifier())
])

# --------------------
# 4. Definir o Grid de parâmetros
# --------------------
param_grid = {
    'scaler': [StandardScaler(), MinMaxScaler(), RobustScaler()],
    'knn__n_neighbors': list(range(1, 21))
}

# --------------------
# 5. Rodar o GridSearch
# --------------------
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

# --------------------
# 6. Visualizar os melhores resultados
# --------------------
results = pd.DataFrame(grid.cv_results_)

# Pega os top 3 pelas maiores médias de acurácia
top3 = results.sort_values(by='mean_test_score', ascending=False).head(3)

print("\nTop 3 melhores configurações:\n")
print(top3[['params', 'mean_test_score']])

# --------------------
# 7. Plotar gráfico de desempenho
# --------------------
plt.figure(figsize=(10, 6))
for index, row in top3.iterrows():
    k_value = row['params']['knn__n_neighbors']
    scaler_name = type(row['params']['scaler']).__name__
    mask = (results['param_knn__n_neighbors'] == k_value) & (results['param_scaler'].apply(lambda x: type(x).__name__) == scaler_name)
    plt.plot(results.loc[mask, 'param_knn__n_neighbors'],
             results.loc[mask, 'mean_test_score'],
             marker='o', label=f'{scaler_name} | k={k_value}')

plt.xlabel('Número de Vizinhos (k)')
plt.ylabel('Acurácia média (cross-validation)')
plt.title('Top 3 Configurações - KNN GridSearch')
plt.legend()
plt.tight_layout()
plt.show()
