from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

concrete_data = pd.read_excel('../Dataset/Concrete_Data.xls')
concrete_data = concrete_data.dropna()
#test = concrete_data[concrete_data.isin([0]).any(axis=1)]
#print(test['Blast Furnace Slag'].value_counts().sort_index()[0])
#print(test['Fly Ash'].value_counts().sort_index()[0])
#print(test['Superplasticizer'].value_counts().sort_index()[0])

X = concrete_data.drop(["Concrete compressive strength","Coarse Aggregate","Fine Aggregate","Fly Ash"], axis=1).values
X_norm = StandardScaler().fit_transform(X)
y = concrete_data["Concrete compressive strength"].values
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

#pd.set_option('display.max_columns', None)
#print(concrete_data.shape)