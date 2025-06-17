from src.utils import  load_df1_unidade2,load_df2_unidade2,load_wine_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils import load_churn_dataset

df1 = load_df1_unidade2()
df2 = load_df2_unidade2()
wine = load_wine_dataset()

scaler = StandardScaler()
X = wine.drop(['Quality'], axis = 1)
X = scaler.fit_transform(wine)
y = wine['Quality'].values
X_train, X_test, y_train,y_test = train_test_split(X,y, stratify = y, test_size = 0.2, random_state = 42)

churn = load_churn_dataset()
print(churn.head(5))





