#Questão 1 – Pré-processamento e Análise Exploratória
#Arquivo: classificacao_q1.py

#Carregue o dataset com pandas.
#Trate valores ausentes.
#Analise a distribuição da variável-alvo.
#Codifique variáveis categóricas, se necessário.
#Faça uma análise estatística exploratória
#Salve o arquivo como classificacao_ajustado.csv.

from src.utils import load_cancer_dataset

cancer = load_cancer_dataset()

#
print(f'Shape do Dataset: {cancer.shape}')

#
print(f'Informações: {cancer.info()}')

#
print(f'Valores Nulos: {cancer.isnull().sum()}')

