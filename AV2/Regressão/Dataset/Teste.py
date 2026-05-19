from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Pre_processamento import *

not_0_data_ = concrete_data.drop(concrete_data[concrete_data['Superplasticizer'] == 0].index)
X2 = not_0_data_.drop("Concrete compressive strength", axis=1).values
y2 = not_0_data_["Concrete compressive strength"].values
X2_bmi = X2[:,4].reshape(-1,1)

regressor = LinearRegression()
regressor.fit(X2_bmi, y2)
pred = regressor.predict(X2_bmi)

plt.boxplot(concrete_data)

sns.heatmap(concrete_data.corr(numeric_only=True), annot=True)
plt.title('Completo')
plt.show()
