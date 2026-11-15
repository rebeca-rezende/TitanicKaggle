# Desafio Kaggle Titanic: Machine Learning from Disaster

Este repositório documenta a minha jornada resolvendo o desafio do Titanic no Kaggle. O objetivo deste projeto foi aplicar os principais conceitos de Ciência de Dados e Machine Learning do zero — desde a limpeza e engenharia de dados até a validação e otimização de modelos preditivos.

---

## Sobre o Desafio
A missão é utilizar informações socioeconômicas e demográficas dos passageiros (como classe, idade, sexo e tamanho da família) para prever quem sobreviveu ou não ao naufrágio. O desempenho é medido através da **acurácia** (porcentagem de acertos).

---

## Organização do Código e Pipeline de Dados
Para organizar o projeto separei o código de pré-processamento em funções reutilizáveis. O fluxo de dados foi construído para blindar o modelo contra *Data Leakage* (vazamento de dados do conjunto de teste para o treino).

### Principais Funções de Pré-processamento (*Data Prep*):
* **`preencher_idade`**: Como cerca de 20% das idades estavam faltando, o preenchimento não foi feito com uma média genérica. Calculamos a mediana agrupando rigorosamente por Classe, Sexo e Porto de Embarque (utilizando estatísticas estritamente extraídas do treino).
* **`identificar_deck`**: A coluna de cabines original tinha muitos valores nulos. Extraímos a primeira letra para identificar o convés (*Deck*) do navio e tratamos os vazios como uma categoria desconhecida (`U`), sabendo que a proximidade com o convés superior afetava as chances de resgate.
* **`preencher_porto`**: Preencheu os raros valores nulos no porto de embarque (`Embarked`) utilizando a moda calculada no conjunto de treino.
* **`preencher_tarifa`**: Substituiu valores de tarifas ausentes utilizando a mediana da respectiva classe de passageiro.
* **`transformacoes_v2` (Engenharia de Recursos):**
  * Conversão binária da coluna de sexo (`male` para 0 e `female` para 1).
  * Criação da métrica `Total_Familiar` (somando irmãos, cônjuges, pais, filhos e o próprio passageiro) e da coluna indicadora `Sozinho`.
  * Extração inteligente de títulos dos nomes (`Mr`, `Mrs`, `Miss`, agrupando títulos raros em `Outros`) utilizando expressões regulares para capturar o status social e a faixa etária.
  * Aplicação de *One-Hot Encoding* (`pd.get_dummies`) para converter variáveis categóricas em colunas numéricas de 0 e 1, garantindo a perfeita simetria estrutural entre o treino e o teste.

---

## A Batalha dos Modelos: O que foi testado
Para avaliar o desempenho, utilizei **Validação Cruzada Estratificada (*StratifiedKFold*) com 5 divisões**, calculando a média das previsões (*Ensemble*) para gerar as submissões. Abaixo estão os 6 testes e modelos que realizei:

1. **Modelo 1 (Random Forest Padrão + K-Fold)**
   * *Abordagem:* Random Forest clássica com parâmetros conservadores sobre as features numéricas básicas.
   * *Desempenho:* **83,16%** na validação local | **78,47%** no Kaggle.
   * *Análise:* O modelo mais equilibrado e estável do projeto, evitando decorar os dados.

2. **Modelo 2 (Random Forest com GridSearchCV + K-Fold)**
   * *Abordagem:* Uso de busca exaustiva de hiperparâmetros combinada com colunas de títulos.
   * *Desempenho:* **82,27%** na validação local | **75,60%** no Kaggle.
   * *Análise:* Sofreu de *overfitting*. O GridSearch ajustou tanto o modelo ao treino que ele perdeu a capacidade de generalizar para dados novos.

3. **Modelo 3 (Random Forest Otimizada com Features Expandidas)**
   * *Abordagem:* Parâmetros manuais mais controlados, mantendo as features numéricas e os títulos principais.
   * *Desempenho:* **83,16%** na validação local | **77,50%** no Kaggle.
   * *Análise:* Recuperou a consistência, mostrando que os títulos ajudam, mas exigem cautela com o ruído.

4. **Modelo 4 (GridSearch Avançado + Threshold Tuning)**
   * *Abordagem:* Busca de hiperparâmetros unificada com a técnica de *Threshold Tuning* (varredura de 40 limiares de probabilidade diferentes para definir o ponto de corte ideal em vez do padrão 0.5).
   * *Desempenho:* **83,84%** na validação local | **78,47%** no Kaggle.
   * *Análise:* Empatou no recorde de submissão, provando que a Random Forest encontrou um teto natural com o conjunto atual de dados.

5. **Modelo 5 (XGBoost Padrão)**
   * *Abordagem:* Migração para algoritmos de *Gradient Boosting* (aprendizado sequencial corretivo).
   * *Desempenho:* **83,50%** na validação local | **75,12%** no Kaggle.
   * *Análise:* Por ser extremamente poderoso, o XGBoost puro decorou os ruídos da base pequena do Titanic, falhando no teste cego.

6. **Modelo 6 (XGBoost Regularizado)**
   * *Abordagem:* O mesmo XGBoost, mas com parâmetros pesados de poda (árvores rasas, penalidades L1/L2 e amostragem reduzida).
   * *Desempenho:* **82,49%** na validação local | **77,99%** no Kaggle.
   * *Análise:* Entregou uma solução altamente robusta e profissional, consolidando uma excelente generalização.

---

## Conceitos de Ciência de Dados Aplicados
Este projeto serviu para consolidar a prática de vários pilares fundamentais da área:
* **Análise Exploratória (EDA):** Identificação de padrões visuais e correlações estatísticas (como o peso determinante do sexo e da classe social na sobrevivência).
* **Engenharia de Recursos (*Feature Engineering*):** Transformação de colunas textuais brutas em variáveis numéricas informativas (títulos e tamanhos de famílias).
* **Validação Robusta (*StratifiedKFold*):** Divisão inteligente da base para evitar métricas falsas de desempenho.
* **Prevenção de Vazamento de Dados (*Data Leakage*):** Garantia de que todas as referências estatísticas (medianas e modas) fossem calculadas exclusivamente no treino e aplicadas de forma isolada.
* **Ensemble Learning:** Combinação de votos/probabilidades de múltiplos modelos para mitigar erros individuais.
* **Otimização e Limiar (*GridSearch & Threshold Tuning*):** Busca por hiperparâmetros e calibração fina do ponto de corte de decisão.
