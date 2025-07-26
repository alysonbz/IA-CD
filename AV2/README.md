# AVALIAÇÃO 2 - Prazo para envio do relatório e código : 26/07/2025
> Orientações para execução da prova.

Esse documento exibe as descrições das questões e a relação dos datasets que devem ser utiizados 
pelos alunos e alunas.

O modelo de documento seguinte mostra como você deve registrar por escrito o desenvolvimento. 
https://docs.google.com/document/d/1hIwPx9W-k3LnXRJrkWYTsbrtx4NfP88_/edit?usp=sharing&ouid=118351454454462119994&rtpof=true&sd=true
  
##  Aluno - Dataset

Daniel Vaz e Kaue Barbosa: 

https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis


Luis Joaquim: 

https://archive.ics.uci.edu/dataset/292/wholesale+customers


Eryka: 

https://archive.ics.uci.edu/dataset/352/online+retail

 
Jefte damasceno e Nivaldo: 

https://archive.ics.uci.edu/dataset/352/online+retail

Luiza e Julia: 

https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

Madson:

https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

Silas e Rick:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

SAMUEL HENRIQUE: 

https://www.kaggle.com/datasets/spscientist/students-performance-in-exams/data

---



#### **1. Análise Exploratória com Foco em Redução de Complexidade**

* Realize a análise exploratória usual (distribuições, correlações, outliers, análise estatística e etc).
* Com base nas correlações e variâncias, **proponha uma estratégia inicial para redução da dimensionalidade.**

  * Justifique se usará normalização, padronização ou transformação específica antes do PCA.
  * Identifique possíveis atributos redundantes.

**O que será avaliado:** clareza da estratégia, uso de argumentos estatísticos, visualizações e pré-análise crítica.

---

#### **2. Redução de Dimensionalidade Múltipla: PCA vs T-SNE**

* Aplique **PCA** e **T-SNE** nos dados.
* Compare os dois métodos sob diferentes aspectos:

  * Preservação da variância ou estrutura
  * Custo computacional
  * Capacidade de visualização de agrupamentos
* Discuta vantagens e desvantagens de cada método no contexto do seu dataset.

**O que será avaliado:** análise comparativa aprofundada, uso correto das técnicas, leitura crítica das projeções.

---

#### **3. Clusterização com K-Means e Hierárquico**

* Aplique dois  métodos de clusterização:

  * **K-Means** (com validação por inertia e silhouette score)
  * **Hierárquico** (com pelo menos dois métodos de ligação – ex: average e complete)
 
* Compare os agrupamentos. 
* Realize visualizações com projeção PCA ou T-SNE para cada método.

**O que será avaliado:** domínio de técnicas diversas, comparação multitécnica, interpretação de resultados e coerência entre visualização e métricas.

---


#### **4. Interpretação Semântica dos Agrupamentos**

* Proponha um método para **nomear ou interpretar semanticamente os clusters**.
  * Escolha o metodo de clusterização adequada.
  * Utilize médias de variáveis por cluster, boxplots comparativos ou análise de centroides.
  * Aplique crosstab para avaliar o alinhamento.
  

**O que será avaliado:** profundidade interpretativa, capacidade de extrair significado prático dos clusters.

---

#### **5.T-SNE ou PCA como Pré-processamento para Classificação**

* Escolha uma variável atributo alvo para um modelo supervisionado.
* Avalie o desempenho da classificação com 3 possíveis cenários:
  * com T-SNE , 
  * PCA 
  * Nomalização apropriada.
* Discuta se a clusterização ajudou a capturar informação relevante para o modelo supervisionado.

**O que será avaliado:** raciocínio analítico, integração de técnicas supervisionadas e não supervisionadas, avaliação de impacto prático.

---


### **Tabela de Avaliação**

| Critério                                          | Scores  |
| ------------------------------------------------- |---------|
| Estratégia de análise e pré-processamento         | 10      |
| Comparação entre PCA e T-SNE                      | 20      |
| Aplicação e comparação crítica de métodos de clusterização | 20      |
| Interpretação semântica e análise dos agrupamentos | 20      |
| Integração com aprendizado supervisionado         | 10      |
| Clareza, justificativas, organização e profundidade analítica | 20       |
| **Total**                                         | **100** |

## Instruções Finais

- Use comentários claros nos scripts e utilize boas práticas de programação.  
- Organize os arquivos conforme a estrutura sugerida.  
- Pode usar a estrutura do Colab para incluir relatório e código no mesmo arquivo.
- Não esqueça de incluir os datasets tratados no repositório.
- Inclua o relatório deNtro da sua branch no github. Não precisa enviar para meu email.

---

**Dúvidas :** alysonbnr@ufc.br   

**Prazo final - para envio do relatório e código:**  26-07-2025

**Prazo final para apresentação em PPT:**  30-07-2025