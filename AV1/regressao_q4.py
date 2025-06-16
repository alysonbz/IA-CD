# 1. Visualize os coeficientes da Lasso.

from sklearn.linear_model import Lasso

X = df2.drop("csMPa", axis=1)
y = df2["csMPa"]

lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X, y)


print("Coeficientes Lasso:", lasso_model.coef_)

##

# 2. Identifique os atributos mais importantes

cof = lasso_model.coef_
features = X.columns

cof = pd.DataFrame({
    "Atributo": features,
    "Coeficiente": cof
})

cof["Importante"] = cof["Coeficiente"].abs()
cof = cof.sort_values(by="Importante", ascending=False)
print(cof[["Atributo", "Coeficiente"]])

##

# 3. Plote um gráfico de importância

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(cof["Atributo"], cof["Importante"])
plt.xlabel("Atributos")
plt.ylabel("Values")
plt.title("Gráfico de importânacia")
plt.tight_layout()

##

# 4. Discuta os resultados.

# Os  resultados mostram o que já se espera da resistência de um concreto que vai depender da qualidade dos materiais, propriamente dito do cimento, aguá e outros métodos como o tempo. O Lasso ajudou a destacar os fatores mais relevantes, o que tornou esse modelo mais eficiente.