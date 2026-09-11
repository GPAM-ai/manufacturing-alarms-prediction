---
tags:
  - IC
data: 2026-09-02
---

# Como lidei com o sigilo no relatório
---
Mantive oculto o nome da empresa parceira e qualquer detalhe relacionado à localização da fábrica. Os dados de alarme e falha estão no GitHub, preciso saber se devem ser removidos antes de deixar público.

Descrevi brevemente o que era cada *feature* do dataset tratado e sobre qual área de produção (algo bem geral) se tratava. Não há uma medida de duração, severidade, causa ou consequência dos alarmes, nem um rótulo de ocorrência de regimes de cascata. Por isso, uma vez que o objeto de estudo da pesquisa é o regime de cascata, ele foi definido como eventos de recorrência rápida — IAT inferior a 30 minutos. Esses fatos impedem uma boa generalização do modelo, mas lidam com uma boa lacuna de pesquisa.
# Pontos Soltos
---
Detalhes da EDA, mudança de target e detalhes da engenharia de feature
- A etapa de limpeza dos dados antes da EDA removeu 3 registros duplicados.
  88.414 -> 88.411 eventos.
- Agregação temporal diária do volume de alarmes gerou uma série temporal de granularidade diária com 1.953 observações. 1.528 com ao menos um evento e 425 dias zerados.
- Uma segunda série temporal foi feita pela identificação de outliers por ano, mantendo a regra do intervalo interquartil (5/95) e substituição por interpolação linear, complementada nas bordas por preenchimento com o valor válido mais próximo. 106 observações foram tratadas. A série resultante não possui dias zerados e nem ultrapassa 150 alarmes/dia. As duas séries foram estudadas na EDA.
- Identificação de hiatos: pelo menos dois dias consecutivos; 32 hiatos identificados. Classificação já havia sido apresentada: COVID-19, paradas programadas, Carnaval, Semana Santa, não classificados e não resolvidos (episódios de 70d e 90d, em jul-out 2022 e ago-nov 2023). Os dias nulos foram interpolados na série de trabalho e considerados ausentes na EDA.
- A análise de recorrência identificou 88.391 intervalos positivos válidos de recorrência entre alarmes para as 20 máquinas. *IAT = Inter-Alarm Time* (intervalo entre alarmes). A distribuição do IAT é assimétrica à direita. As medianas para as máquina pairam entre 30 a 100 minutos de intervalo. As máquinas TR19, TR10 e TR17 foram as que demonstraram menor IAT.
	- Em todas as máquinas, ao menos 25% dos registros de intervalo são recorrências de até 15min; no limiar de 60min, a proporção mínima supera 35%. O limiar de 30min foi escolhido pelo equilíbrio entre sensibilidade a repetições rápidas e abrangência operacional. Neste último limiar estão incluídas 48%, 48% e 39% das recorrências das máquinas TR19, TR10 e TR17, respectivamente. As proporções mais inferiores são 29,92% para TR14 e TR18, evidenciando que a repetição rápida não se restringe a poucas máquinas.
- Análise espectral revelou forte potência espectral na faixa semanal, seja com ou sem outliers. Faixa de poucos dias teve alta potência na série bruta em razão dos valores extremos, mas não é uma relação causal simples, pois o valor continuou alto na série tratada.
- Decomposição Aditiva foi feita manualmente (com base em código desenvolvido em RPVMM II). Implementação conta com uma média móvel centrada de 30 dias para estimar a tendência. Mostrar imagem. Na série bruta, a tendência revela níveis mais elevados em 2020 e 2021, oscilações em 2022, forte redução em 2023 e recuperação a partir de 2024. Na série sem outliers, a amplitude semanal foi 1.50, demonstrando uma maior estabilidade; a forma geral da tendência foi preservada. As mudanças de longo prazo não eram produzidas por outliers
- Autocorrelação de até 90 dias apresentou coeficientes positivos que decaem lentamente, permanecendo acima do IC-95%. Isso evidencia memória temporal, e pode refletir tendência, mudanças de regime e longas sequências de zeros. Os lags de 1 a 5, 7 e 14 se mantiveram os mais consistentes nas análises.
	- Análise anual demonstrou heterogeneidade temporal, o que sugere estrutura temporal não estacionária da série, hipótese que foi testada durante modelagem.
- Correlação com variáveis exógenas (com e sem outliers) revelou que Selic e IPCA-Duráveis eram as variáveis exógenas mais correlatas de forma consistente.
- Feature-alvo: 
- Conjunto simples final:
- Conjunto completo final:
- Os dois conjuntos somaram 1.499 registros no total.

# Implementação dos Modelos
---
Descrever quais modelos escolhi e por quê. Comentar novo método de validação proposto.

# Resultados Finais e Contribuições
---
Comentar os resultados finais dos modelos, destacando o XGBoost como o melhor e qual contribuição científica/industrial esta pesquisa trouxe.

# Artigo para Publicação
---
Identificar o que é relevante utilizar para construir um artigo **em inglês** para submissão em Congressos e Revistas. Propor isso para o Didier. Pesquisar qual modelo usa o evento que o Didier comentou no Pumble e desenvolver um protótipo.

Análises feitas; focar em análises de falha em cascata de alarmes. Revisão da literatura; regressão de volume diário de alarmes; empresa automotiva; treinamento de MLP. Artigo seria revisão da literatura atualizada, pré-processamento, análise de dados, engenharia de features e desenvolvimento e validação dos modelos. Terminar com conclusões, não preciso aprofundar tanto contribuição científica, é mais discussão de resultados. Contribuição é análise de dados reais. Proposta é um framework para analisar falha em cascata. Tentar LSTM com GridSearch e MLP com Optune. Analisar correlação com variáveis macroeconômicas.
- (Sab) Revisão de literatura da LSTM e MLP com teste de hiperparâmetros
- (Seg) Implementar essas redes no conjunto de dados
- (Seg) Sumário do paper
	- Introduction, related works, material and methods, methodology (desenvolvimento de modelos, dataset, figuras, pré processamento), experimental results, final remarks (discussão e conclusão), aknowledgements (agradecimento) e referências
- (Qua) Primeira versão final - escrita em inglês

# Ideias de TCC
---
Manter contato com o Didier e comentar sobre ideias para TCC. Preciso pensar bem em ideias próprias e boas. Isso poderia ser uma reunião a parte. Usar série de recorrência e séries temporais.

Encontrar datasets benchmark, lidar com programação agêntica, métodos de processamento de dados para redes neurais, programação por contexto.