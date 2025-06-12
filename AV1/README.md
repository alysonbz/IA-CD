# AVALIAÇÃO 1
> Orientações para execução da prova.

Esse documento exibe as descrições das questões e a relação dos datasets que devem ser utiizados 
pelos alunos e alunas.

O modelo de documento seguinte mostra como você deve registrar por escrito o desenvolvimento. 
https://docs.google.com/document/d/1hIwPx9W-k3LnXRJrkWYTsbrtx4NfP88_/edit?usp=sharing&ouid=118351454454462119994&rtpof=true&sd=true

##  Aluno - Dataset

Daniel Vaz e Kaue Barbosa: 

A: https://www.kaggle.com/datasets/erdemtaha/cancer-data 

B: https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset



Luis Joaquim: 

A: https://www.kaggle.com/datasets/praveengovi/credit-risk-classification-dataset

B: https://www.kaggle.com/datasets/schirmerchad/bostonhoustingmlnd

Alana e João Santiago: 

A: https://www.kaggle.com/datasets/primaryobjects/voicegender

B: https://www.kaggle.com/datasets/mirichoi0218/insurance

Eryka: 

A: https://www.kaggle.com/datasets/uciml/mushroom-classification

B:  https://www.kaggle.com/datasets/aungpyaeap/fish-market

 

Jefte damasceno e Nivaldo: 

A: https://www.kaggle.com/datasets/rtatman/chocolate-bar-ratings

B:  https://www.kaggle.com/datasets/hellbuoy/car-price-prediction

Luiza e Julia: 

A: https://www.kaggle.com/datasets/prathamtripathi/drug-classification

B:  https://www.kaggle.com/datasets/uciml/student-alcohol-consumption


Madson:

A: https://www.kaggle.com/datasets/whenamancodes/predict-diabities

B: https://www.kaggle.com/datasets/maajdl/yeh-concret-data


Silas e Rick:

A: https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

B: https://www.kaggle.com/datasets/quantbruce/real-estate-price-prediction

SAMUEL HENRIQUE: 

A: https://www.kaggle.com/datasets/praveengovi/credit-risk-classification-dataset?select=customer_data.csv

B: www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data

---

## Parte A – Classificação

### Questão 1 – Pré-processamento e Análise Exploratória  
**Arquivo:** `classificacao_q1.py`

1. Carregue o dataset com pandas.  
2. Trate valores ausentes.  
3. Analise a distribuição da variável-alvo.  
4. Codifique variáveis categóricas, se necessário.  
5. Faça uma análise estatística exploratória
5. Salve o arquivo como `classificacao_ajustado.csv`.

---

### Questão 2 – KNN Manual com Várias Distâncias  
**Arquivo:** `classificacao_q2.py`

1. Divida o dataset em treino e teste.  
2. Implemente manualmente o KNN.  
3. Avalie usando:
   - Euclidiana
   - Manhattan
   - Chebyshev
   - Mahalanobis  
4. Compare os resultados obtidos para os diferentes valores de distancia, considerando a métrica acurácia.

---

### Questão 3 – Normalização, alteração do valor de K e Efeito no KNN  
**Arquivo:** `classificacao_q3.py`

1. Aplique as normalizações: logarítmica, minMax, StandardScaler.  
2. Reaplique o KNN manual com a melhor distância.  
3. Compare os desempenhos com e sem normalização.
4. Faça também uma análise manual para descobrir o melhor valor de K para o KNN. Plote o gráfico de acurácia de treino e teste versus `k`.

---

### Questão 4 – KNN com `sklearn` + GridSearch  
**Arquivo:** `classificacao_q4.py`

1. Use `KNeighborsClassifier` da Sklearn.  
2. Teste valores de `k` entre 1 e 20 e as normalizações com `GridSearchCV`.  
3. Plote gráfico dos resultados com as 3 melhores configurações.  


---

### Questão 5 – Cross-Validation e Avaliação Final  
**Arquivo:** `classificacao_q5.py`

1. Aplique `cross_val_score` (5 folds) com a melhor configuração possível que você determinou do seu classificador.  
2. Exiba: média, desvio padrão, matriz de confusão e `classification_report`.  
3. Interprete os resultados quantitativamente.

---

## Parte B – Regressão

### Questão 1 – Pré-processamento e Correlação  
**Arquivo:** `regressao_q1.py`

1. Carregue o dataset.  
2. Trate valores ausentes.  
3. Verifique a correlação com a variável-alvo.  
4. Normalize ou padronize os dados.  
5. Salve como `regressao_ajustado.csv`.

---

### Questão 2 – Regressão Linear Simples  
**Arquivo:** `regressao_q2.py`

1. Aplique Regressão Linear (`sklearn`).  
2. Plote a reta de regressão (feature mais correlacionada).  
3. Calcule RMSE e R².  
4. Comente sobre os resultados.

---

### Questão 3 – Linear vs Ridge vs Lasso  
**Arquivo:** `regressao_q3.py`

1. Aplique as regressões: Linear, Ridge e Lasso.  
2. Use `cross_val_score` com 5 folds.  
3. Compare RMSE e R² em uma tabela.  
4. Identifique o melhor modelo.

---

### Questão 4 – Coeficientes e Seleção de Atributos  
**Arquivo:** `regressao_q4.py`

1. Visualize os coeficientes da Lasso.  
2. Identifique os atributos mais relevantes.  
3. Plote um gráfico de importância.  
4. Discuta os resultados.

---

## Instruções Finais

- Use comentários claros nos scripts e utilize boas práticas de programação.  
- Organize os arquivos conforme a estrutura sugerida.  
- Não esqueça de incluir os datasets tratados no repositório.
- Inclua o relatório detro da sua branch no github. Não precisa enviar para meu email.

---

**Dúvidas :** alysonbnr@ufc.br   

**Prazo final - para envio do relatório e código:**  16-06-2025

**Prazo final para apresentação em PPT:**  18-06-2025
