from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Pre_processamento import *

from sklearn.preprocessing import StandardScaler
names = concrete_data.drop(["Concrete compressive strength"], axis=1).columns
total = concrete_data.drop(["Concrete compressive strength"], axis=1).values

lasso = Lasso()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(total)
lasso.fit(X_scaled, y)
reg_coef = lasso.coef_
plt.bar(names, reg_coef)
plt.ylabel('Coeficiente de Regressão')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8,5))
sns.heatmap(concrete_data.corr(numeric_only=True), annot=True)
plt.title('Completo')
plt.show()

X_plt = total[:,0].reshape(-1,1)
regressor = LinearRegression()
regressor.fit(X_plt,y)
pred = regressor.predict(X_plt)
plt.title(label=f"Regressão linear Cimento")
plt.scatter(X_plt,y)
plt.plot(X_plt,pred, color='red')
plt.show()

#plt.boxplot(concrete_data)
#plt.show()
