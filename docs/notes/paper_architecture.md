---
tags:
  - IC
  - UNIFESP
data: 2026-09-10
---
# Detalhes de Implementação
---
Esta seção documentará como será desenvolvido os experimentos com MLP e LSTM no atual ambiente. A nota [[paper_new_experiments]] já define alguns detalhes da arquitetura que devem ser seguidos.

Devem ser utilizados os mesmos conjuntos de atributos já definidos anteriormente (simples - 16 atr. / completo - 36 atr.). Os *folds* expansivos serão utilizados da mesma forma que nos treinos com XGBoost e Random Forest. Deve-se utilizar uma grade pequena variando os hiperparâmetros como sugerem as referências. A implementação do Optuna também deve ser de acordo com as referências definidas. Utilizar biblioteca PyTorch. Pesquisar por referências no GitHub de implementações desses *frameworks* para ser utilizados como base. 

Métricas de validação:

- todas as utilizadas nos experimentos com XGBoost e Random Forest;
- MAE médio é a principal;
- registro das melhores configurações de cada experimento;
- registro do tempo gasto para computação ("rodar") cada experimento;

Processo de validação e síntese de resultados: interpretar resultado de cada modelo (comentar suas conclusões, suas vantagens para pesquisa e onde o modelo se limitou); levantar conclusões sobre o quanto a qualidade dos dados fabris impactaram nos resultados e quais limitações a natureza desses dados trouxe. Comparar métricas obtidas em cada experimento e comentar em qual delas cada modelo ganha ou perde. Relação custo-benefício entre MAE e tempo de execução é elemento chave para avaliar qual se saiu melhor.