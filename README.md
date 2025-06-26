**Análise de Risco Gestacional (MVP)**

**Visão Geral do Projeto**

Este projeto tem como objetivo principal explorar as características de saúde de mulheres gestantes para identificar indicadores de risco durante a gravidez. Através da análise de dados cruciais como idade, pressão arterial, glicemia, temperatura corporal e frequência cardíaca, buscamos compreender a classificação de risco gestacional associada e preparar esses dados para futuras aplicações de Machine Learning.

**Dataset**

O conjunto de dados utilizado, denominado 'Pregnancy', foi compilado para identificar riscos na saúde materna.
Você pode encontrar o dataset original aqui: https://github.com/AlineFiori/MVP/raw/refs/heads/main/Maternal%20Health%20Risk%20Data%20Set.csv.xls


**Configuração do Projeto**

Para rodar a análise, abra o Template_Analise_de_Dados_Pregnancy.ipynb em um ambiente do Google Colab, e execute as células sequencialmente.

É necessário instalar as seguintes bibliotecas para este projeto:

- Manipulação e análise de dados: import pandas as pd
- Computação numérica e científica em Python: import numpy as np
- Plotagem e visualização de dados: import matplotlib.pyplot as plt
- Visualização de dados científicos: import seaborn as sns
- Utilizado na valiação de modelos de Machine Learning, divide o dataset em treino e teste: from sklearn.model_selection import train_test_split
- Para escalonamento de dados (normalização): from sklearn.preprocessing import MinMaxScaler
- Para padronizacao de dados (média de 0 e desvio padrão de 1): from sklearn.preprocessing import StandardScaler


**Metodologia e Análise Realizada**

O trabalho foi estruturado em diversas etapas, garantindo uma análise robusta e a preparação adequada dos dados.

1. Analise Exploratória
   
   1.1 Estrutura do Dataset: A etapa de configuração e adequação dos dados é crucial para a análise de dados e Machine Learning, pois garante a qualidade do modelo, a             interpretabilidade dos resultados, a eficiência do processo e a consistência do formato dos dados.

   1.2 Estatística Descritiva: A estatística descritiva é uma fase essencial que organiza e resume os dados estatisticamente para torná-los compreensíveis, permitindo a           identificação de problemas e anomalias, guiando as decisões de pré-processamento e revelando insights iniciais sobre as características do conjunto de dados.

   1.3 Análise de Variáveis Numéricas e categóricas: É fundamental para extrair insights iniciais, validar hipóteses e guiar as próximas etapas do desenvolvimento do projeto.     Entender a natureza e o comportamento de cada tipo de variável permite construir uma base sólida para modelos mais complexos e decisões estratégicas.

   1.4 Matriz de Correlação: É uma ferramenta de suma importância, quase indispensável, para compreender as relações entre as variáveis numéricas do conjunto de dados. Ela        serve como um mapa rápido e eficiente para identificar padrões, dependências e possíveis problemas antes de mergulhar em análises mais complexas ou na construção de            modelos.As correlações positivas indicam que as variáveis tendem a se mover na mesma direção.

  
2. Pré Processamento de Dados: O pré-processamento de dados é uma etapa crucial para preparar os dados para modelagem, ou seja, transformar os dados brutos em um formato adequado para algoritmos de aprendizado de máquina, garantindo que estejam no formato correto e otimizados para o desempenho do algoritmo.

  2.1 Treino e Teste: Esta etapa consiste em separar o dataset em duas partes, sendo um conjunto de treino e outro de teste. O conjunto de treino é usado para "ensinar" o        modelo a encontrar padrões nos dados. O conjunto de teste é um conjunto de dados "invisível" que o modelo nunca viu durante o treinamento. Ele é usado para simular como o      modelo se comportaria com novos dados do mundo real, dando uma estimativa imparcial de sua capacidade de generalização.
  
  2.2 Normalização: A normalização é fundamental para garantir que as análises e os modelos iniciais sejam justos, precisos e interpretabis, especialmente quando se lida com     variáveis em diferentes escalas. Escala os dados para um intervalo fixo, geralmente entre 0 e 1. É útil quando o algoritmo de machine learning assume que as características    estão em uma escala semelhante.
  
  2.3 Padronização: A padronização (ou Z-score scaling) transforma os dados para ter média 0 e desvio padrão 1. Isso garante que todas as features contribuam igualmente para o   modelo, evitando que aquelas com valores maiores dominem as que têm valores menores.

  2.4 Outras etapas de pré processamento: Nesta etapa comento um pouco sobre a possibilidade de aplicar o PCA, o ganho e as desvantagens que este método apresenta sobre este     trabalho.
  
3. Conclusão
O objetivo principal deste MVP foi explorar as características de saúde em mulheres gestantes para identificar indicadores de risco e preparar os dados para futuras etapas de modelagem preditiva. Através de diversas etapas de análise, insights importantes foram obtidos:

Preparação e Engenharia de Features:
- Enriquecimento do Dataset: A criação da coluna Idade_35+ foi uma decisão estratégica baseada em conhecimento de domínio médico, reconhecendo que a idade gestacional a partir de 35 anos é um fator de risco comprovado. Esta feature binária é um preditor valioso que captura uma relação não-linear com o risco.
- Codificação da Variável Alvo: A transformação da variável categórica Nivel_Risco em Nivel_Risco_Numerico (0=low risk, 1=mid risk, 2=high risk) foi uma etapa fundamental e necessária, convertendo os dados para um formato compreensível pelos algoritmos de Machine Learning.
- Padronização dos Dados: A aplicação do StandardScaler nos dados de treino garantiu que todas as features numéricas tivessem média zero e desvio padrão um. A análise confirmou que esta etapa foi executada corretamente, mantendo as relações intrínsecas entre as variáveis (evidenciado pela estabilidade da matriz de correlação) e preparando o dataset para modelos sensíveis à escala das features.

Análise de Correlação e Fatores de Risco:
A matriz de correlação revelou insights cruciais sobre a associação linear entre as características de saúde e o nível de risco gestacional:

Principais Preditores de Risco: 
O Nível de Glicose demonstrou a associação positiva mais forte com o risco gestacional (0.5777), indicando que níveis mais altos de glicose estão fortemente ligados a um risco elevado. As Pressões Sistólica (0.3771) e Diastólica (0.3372) também se mostraram preditores importantes, com correlações positivas moderadas. Impacto da Idade: Tanto a Idade (0.2919) quanto a Idade_35+ (0.2953) apresentaram correlações positivas moderadas com o nível de risco, confirmando que a idade é um fator relevante.

Multicolinearidade: 
Foi identificada alta multicolinearidade entre Idade e Idade_35+ (0.8546), e entre Pressao_Sistolica e Pressao_Diastolica (0.7993). Embora isso não seja um problema para todos os modelos (ex: árvores de decisão), pode exigir atenção para modelos lineares, onde técnicas como PCA ou seleção de features podem ser consideradas.


Respostas às hipóteses levantadas, em relação a idade gestacional 35+ (gestante com 35 anos ou mais):
- Qual o percentual deste grupo de risco na amostra avaliada? No dataset avaliado, 31,36%% das gestantes pertencem ao grupo de idade maior ou igual a 35 anos (Idade_35+). Este é um grupo significativo na amostra, o que sublinha a importância de analisar seus perfis de risco.

- Qual o percentual de gestação de alto risco? Dentro do nível de alto risco gestacional (righ risk), 48,4% são gestantes 35+. Um valor bastante significativo.

- Qual a correlação entre a idade materna e o risco gestacional? A correlação variou pouco entre os dados originais, normalizados e padronizados, como era esperado. Em todos eles a correlação se mostrou relativamente fraca a moderada. Isso significa que, à medida que a idade da gestante aumenta, há uma tendência de o risco gestacional também aumentar, embora a força da correlação sugere que a idade por si só não explica uma grande parte da variação no nível de risco, outros fatores também contribuem para aumentar o risco, como por exemplo, diabetes gestacional que possui correlação positiva mais forte com o nível de risco.

Próximos Passos:
Com os dados limpos, transformados e as relações iniciais compreendidas, o próximo passo lógico e crucial é a construção e avaliação de modelos de Machine Learning. Este MVP estabeleceu uma base sólida e insights valiosos para o desenvolvimento de um sistema preditivo robusto para a avaliação de risco gestacional.
Este trabalho estabelece uma base sólida para o desenvolvimento de um sistema preditivo robusto na avaliação de risco gestacional.
