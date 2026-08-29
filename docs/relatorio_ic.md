# 1. Introdução

## 1.1 Contextualização e Motivação

A Indústria 4.0 ampliou a disponibilidade de dados operacionais e a aplicação de métodos preditivos na manufatura. Entre essas aplicações, a manutenção preditiva busca antecipar condições anormais e orientar intervenções antes que ocorram interrupções de produção. Em linhas automotivas altamente automatizadas, essa necessidade é especialmente relevante porque alarmes sucessivos podem sobrecarregar os operadores, dificultar a identificação da causa inicial e comprometer a continuidade do processo.

Esta pesquisa investiga registros reais de alarmes emitidos por máquinas de soldagem da área de *Body-in-White* (BiW) de uma planta automotiva parceira, entre janeiro de 2020 e maio de 2025. O trabalho foi desenvolvido no PIBITI da Universidade Federal de São Paulo (UNIFESP), campus São José dos Campos, em parceria com pesquisa dirigida pela FAPESP (proc. 2020/09850-0). Busca-se caracterizar os padrões temporais desses eventos e prever a intensidade diária de recorrências rápidas. Como a base não contém um rótulo de cascata, adota-se como variável-alvo o número diário de alarmes ocorridos menos de 30 minutos após o evento anterior da mesma máquina. Essa medida funciona como uma *proxy* quantitativa de concentração temporal, sem pressupor propagação causal entre equipamentos.

## 1.2 Objetivos

### Objetivo Geral

Desenvolver uma metodologia reproduzível de análise e modelagem de alarmes industriais para prever a quantidade diária de recorrências com IAT inferior a 30 minutos, produzindo um modelo que possa apoiar o planejamento da manutenção na planta parceira.

### Objetivos Específicos

Para alcançar esse objetivo, foram definidos os seguintes objetivos específicos:

1. Caracterizar a estrutura, a distribuição temporal e as limitações do *dataset*.
2. Diagnosticar os hiatos da série e avaliar seu efeito sobre a continuidade dos dados.
3. Investigar recorrências por máquina e turno de produção.
4. Identificar tendências, sazonalidades e anomalias temporais na série diária.
5. Avaliar a associação dos alarmes com variáveis macroeconômicas candidatas.
6. Analisar a autocorrelação e selecionar defasagens potencialmente preditivas.
7. Construir um conjunto simples e outro completo de *features* a partir dos resultados da análise exploratória.
8. Comparar modelos de regressão, analisar a importância das *features* e selecionar uma solução compatível com a aplicação industrial.

## 1.3 Estrutura do Relatório

Após esta introdução, a Seção 2 reúne os fundamentos e trabalhos relacionados; a Seção 3 descreve os dados; a Seção 4 apresenta a análise exploratória; a Seção 5 trata da engenharia de *features*; e a Seção 6 expõe a modelagem e os resultados. As Seções 7 e 8 discutem as contribuições, implicações, limitações, conclusões e trabalhos futuros.

# 2. Revisão da Literatura

## 2.1 Manutenção Preditiva e Alarmes Industriais

A manutenção preditiva utiliza dados operacionais para apoiar intervenções antes da ocorrência de eventos críticos. Sua aplicação depende da qualidade dos dados, de uma engenharia de *features* adequada e da seleção de modelos compatíveis com o processo industrial **[Achouch et al., 2022; Carvalho et al., 2019; Hector & Panjanathan, 2024]**. Apesar do avanço da área, estudos baseados em dados reais de plantas em operação ainda são menos frequentes que experimentos com dados simulados ou laboratoriais **[Zonta et al., 2020]**.

Segundo a IEC 62682, um alarme indica uma condição anormal que requer resposta do operador **[IEC 62682, 2022]**. Quando muitos eventos ocorrem em sequência, o volume pode dificultar a priorização e a identificação da origem do problema, situação relacionada a *alarm floods* e regimes de cascata **[ISA 18.2, 2009]**. Como os dados deste trabalho não identificam diretamente tais regimes, a concentração temporal é representada pelo número diário de recorrências com IAT inferior a 30 minutos.

## 2.2 Séries Temporais e Modelagem Preditiva

Séries temporais permitem representar a evolução de processos industriais e investigar tendências, sazonalidades, descontinuidades e dependências entre observações. Em manutenção preditiva, essas propriedades podem ser convertidas em variáveis de calendário e defasagens do histórico operacional, desde que a validação preserve a ordem cronológica dos dados **[Kashpruk et al., 2023]**.

O *Random Forest* combina árvores de decisão treinadas sobre diferentes amostras, enquanto o *XGBoost* constrói árvores sequencialmente para corrigir os erros das anteriores **[Breiman, 2001; Chen & Guestrin, 2016]**. Ambos são adequados a dados tabulares, capturam relações não lineares e fornecem medidas de importância das variáveis. Por essas características, foram comparados neste trabalho nos conjuntos simples e completo de *features*.

## 2.3 Trabalhos Relacionados e Lacuna de Pesquisa

Trabalhos anteriores discutem a aplicação de aprendizado de máquina à manutenção preditiva, a comparação entre desempenho e interpretabilidade e a integração de modelos aos sistemas industriais **[Susto et al., 2015; Achouch et al., 2022]**. Entretanto, há pouca literatura dedicada à previsão de concentrações de alarmes com dados reais fabris, especialmente quando não existe rótulo explícito de cascata. Esta pesquisa contribui nesse espaço ao documentar a preparação, a análise temporal e a modelagem de uma base real da planta parceira, preservando suas limitações operacionais.

# 3. Dados

## 3.1 Contexto e Descrição do *Dataset*

A área *Body-in-White* corresponde à etapa da fabricação automotiva em que componentes metálicos são unidos para formar a carroceria antes da pintura e da montagem final. Por concentrar operações automatizadas de soldagem, sua continuidade depende do funcionamento coordenado das máquinas; alarmes recorrentes podem indicar condições que demandam atenção da equipe de manutenção.

O arquivo disponibilizado pela empresa parceira contém 88.414 registros de alarmes entre 11 de janeiro de 2020 e 16 de maio de 2025. Após a remoção de três duplicatas, foram mantidos 88.411 eventos de 20 máquinas, distribuídos entre as linhas TDF, com 50.368 registros, e TLF, com 38.043. A agregação diária cobre 1.953 dias: 1.528 apresentam pelo menos um evento e 425 não possuem alarmes registrados.

Cada linha representa um evento e contém o instante de ocorrência, a linha, a máquina, o segmento operacional (*trato*), a *bilancella*, a posição e três códigos de alarme, além de dois campos constantes de identificação da área e da estação. O *timestamp* e a máquina permitem ordenar os eventos, calcular o IAT e construir a variável-alvo. Para a modelagem final, os demais campos operacionais não foram utilizados por falta de evidência de relevância e de documentação semântica suficiente.

## 3.2 Limitações dos Dados

A base não informa duração, severidade, causa ou consequência operacional dos alarmes. Também não contém um rótulo de cascata; por isso, o alvo baseado em IAT inferior a 30 minutos mede recorrência rápida, mas não comprova relação causal entre eventos ou impacto sobre a produção.

Os dias sem registros não podem ser interpretados automaticamente como operação normal. A série contém paradas conhecidas e hiatos sem causa validada, incluindo intervalos de 70 e 90 dias. Esses períodos são mantidos como informação de indisponibilidade e tratados explicitamente por indicadores de qualidade na engenharia de *features*.

Por fim, os dados provêm de uma única planta e de duas linhas da área BiW. Os padrões encontrados podem depender das máquinas, do calendário e das práticas de registro locais, de modo que a generalização para outros ambientes exige nova validação.

# 4. Análise Exploratória de Dados

## 4.1 Estratégia Metodológica e Ferramentas Utilizadas

A análise exploratória de dados conduzida foi concebida como uma investigação sistemática e progressiva, estruturada em torno de um princípio metodológico central: partir da caracterização temporal global da série de alarmes para, gradualmente, decompor o comportamento do sistema em suas dimensões operacionais constituintes — máquina, turno de produção e correlação com variáveis exógenas. Essa progressão geral para especificação reflete a hierarquia natural dos fenômenos investigados, na qual padrões macrotemporais, como sazonalidades e tendências, precisam ser identificados e isolados antes que comportamentos de máquinas individuais possam ser devidamente interpretados sem confusão.

O ponto de partida foi a transformação do dado bruto em representação analítica adequada. O *dataset* original consiste em eventos discretos de alarme — registros individuais — que, por sua natureza pontual, não são diretamente compatíveis com os métodos de análise de séries temporais contínuas. A solução adotada foi a agregação diária dos eventos por volume global, produzindo uma série temporal de contagens com granularidade de um dia. Essa representação foi empregada no diagnóstico de hiatos, na análise espectral, na decomposição aditiva, na autocorrelação e na associação com variáveis exógenas. As análises que dependem da identidade da máquina, do horário ou da ordem exata dos eventos — sazonalidade intradiária, IAT e recorrência — utilizaram os registros tratados em granularidade original, sem reconstrução de uma segunda base.

Todo o desenvolvimento analítico foi realizado em Python 3.13, em arquivos de código modular combinado com *notebooks Jupyter*, oferecendo a combinação de executabilidade limpa e documentação narrativa que garantem boa reprodutibilidade. O código completo, organizado em módulos correspondentes a cada etapa da análise encontra-se disponível em repositório público no GitHub [REF], conferindo transparência e rastreabilidade metodológica de trabalho.

As bibliotecas computacionais utilizadas foram selecionadas com base em adoção consolidada na comunidade científica e adequação às tarefas específicas de cada etapa da análise. A biblioteca *Pandas* constituiu a estrutura central de manipulação e indexação temporal dos dados, por sua capacidade nativa de operar com índices de data e hora e realizar reestruturações e agregações temporais de forma eficiente; operações numéricas auxiliares foram realizadas com *NumPy*; a análise espectral foi implementada com os módulos `scipy.fft` e `scipy.signal` da biblioteca *SciPy*; os coeficientes de correlação de Spearman foram obtidos com `scipy.stats`; e as funções de autocorrelação e autocorrelação parcial foram computadas com *Statsmodels*. A decomposição utilizada no resultado final foi implementada de forma modular no próprio projeto, mediante média móvel centrada de 30 dias e estimação sequencial das componentes semanal e mensal. As visualizações foram produzidas com Matplotlib e Seaborn.

A adoção de uma pilha computacional inteiramente baseada em software livre e de código aberto assegura que os resultados obtidos sejam integralmente reproduzíveis por pesquisadores externos sem dependência de licenças proprietárias, alinhando-se aos princípios de ciência aberta e replicabilidade metodológica **[REF]**.

## 4.2 Pré-Processamento e Agregação Temporal

O *dataset* bruto, composto por eventos discretos de alarme com carimbo de tempo individual, foi transformado em uma série temporal contínua apta à análise estatística por meio de um pipeline de pré-processamento implementado em Pandas.

A etapa inicial compreendeu a padronização dos nomes e tipos de dados — com conversão explícita do campo *Timestamp* para o formato temporal do Pandas —, a remoção de três registros duplicados, a eliminação de valores ausentes remanescentes e a ordenação cronológica. A agregação foi realizada na granularidade diária por reamostragem do índice temporal, contabilizando o volume global de alarmes. Em seguida, o índice foi expandido para cobrir continuamente o intervalo de 11 de janeiro de 2020 a 16 de maio de 2025, com preenchimento por zero nos dias sem registros, tornando os hiatos explicitamente representados na série.

O resultado é uma série temporal diária bruta com 1.953 observações, das quais 1.528 possuem ao menos um evento e 425 apresentam contagem zero. Uma segunda versão foi construída pela identificação anual de outliers mediante a regra do intervalo interquartil e sua substituição por interpolação linear, complementada nas bordas por preenchimento com o valor válido mais próximo. Ao todo, 106 observações foram tratadas, e a série resultante não contém valores ausentes nem ultrapassa aproximadamente 150 alarmes por dia. A série bruta permaneceu preservada como referência; sazonalidade diária, FFT, decomposição aditiva e autocorrelação foram comparadas nas versões com e sem outliers.
## 4.3 Diagnóstico e Classificação de Hiatos Temporais

A presença de períodos de zero alarme na série temporal agregada constitui um desafio metodológico relevante: hiatos introduzem descontinuidades que podem distorcer estimativas de sazonalidade, inflar medidas de autocorrelação e comprometer a validade de modelos treinados sob premissa de continuidade temporal. O diagnóstico sistemático desses períodos foi, portanto, conduzido antes das análises espectral e de correlação.

A identificação foi realizada pela detecção de todas as sequências contíguas de dias com contagem zero na série reindexada. O inventário completo reúne 48 sequências; para a análise de relevância operacional, adotou-se o limiar mínimo de dois dias consecutivos, resultando em 32 hiatos. Essa distinção preserva os episódios de um dia para rastreabilidade, mas impede que ausências isoladas recebam o mesmo peso analítico de paralisações prolongadas.

Os 32 hiatos com pelo menos dois dias foram classificados a partir de intervalos de datas previamente anotados: um período de 41 dias associado à paralisação da COVID-19; três paradas de planta, que totalizam 81 dias; três recessos de Carnaval; quatro intervalos associados à Semana Santa; 19 hiatos sem classificação específica; e dois episódios não resolvidos de 70 e 90 dias, que somam 160 dias. Estes últimos ocorreram entre julho e outubro de 2022 e entre agosto e novembro de 2023 e requerem validação da empresa parceira antes que lhes seja atribuída causa operacional. Nas estimativas de FFT e decomposição, os dias de contagem zero foram tratados como ausentes e interpolados apenas na série de trabalho, mantendo-se os valores originais inalterados na base. Esse procedimento reduz a influência espectral das paralisações sem eliminar a informação de que elas ocorreram.
## 4.4 Análise de Recorrência por Máquina e Turno

A recorrência foi investigada por meio do IAT (*inter-alarm time*), definido como o intervalo, em minutos, entre um alarme e o evento imediatamente anterior da mesma máquina. Para seu cálculo, os 88.411 registros tratados foram ordenados por máquina e instante de ocorrência, e aplicou-se a diferença temporal dentro de cada grupo. O primeiro evento de cada máquina não possui observação anterior e, portanto, não gera IAT; intervalos nulos ou negativos seriam igualmente desconsiderados. O procedimento produziu 88.391 intervalos positivos válidos para as 20 máquinas.

A distribuição do IAT é fortemente assimétrica à direita, com médias muito superiores às medianas em razão das paralisações e dos grandes hiatos. Por esse motivo, a mediana é a medida mais adequada para comparar a frequência típica entre máquinas. TR19 apresentou o menor IAT mediano, 38,71 minutos, seguida por TR10, com 43,77 minutos, e TR17, com 73,08 minutos. Esses resultados diferenciam TR19 do simples ranking por volume: apesar de registrar 4.957 alarmes, abaixo de TR10 e TR17, ela concentra eventos em intervalos particularmente curtos.

Foram avaliados limiares de 15, 30 e 60 minutos. Em todas as máquinas, ao menos cerca de 25% dos alarmes pertencem a recorrências inferiores a 15 minutos; no limiar de 60 minutos, a proporção mínima supera aproximadamente 35%. O limiar intermediário de 30 minutos oferece equilíbrio entre sensibilidade a repetições rápidas e abrangência operacional, razão pela qual foi escolhido para a definição da variável-alvo. Nesse limiar, TR19 apresentou 2.389 recorrências em 4.957 alarmes (48,19%), TR10 apresentou 3.036 em 6.379 (47,59%) e TR17 apresentou 2.146 em 5.527 (38,83%). Na extremidade inferior, TR14 e TR18 registraram 29,92%, ainda evidenciando que a repetição rápida não se restringe a poucas máquinas.

Para a análise por turno, cada evento foi associado a uma das três janelas de oito horas: primeiro turno, das 6h às 14h; segundo turno, das 14h às 22h; e terceiro turno, das 22h às 6h. Foram contabilizados 33.957, 30.159 e 24.295 alarmes, respectivamente. As taxas de recorrência abaixo de 30 minutos são geralmente maiores no primeiro e no segundo turnos e menores no terceiro, com destaque para TR10, que alcançou 49,5%, 53,0% e 35,9%, e TR19, com 49,5%, 48,2% e 46,3%. O resultado descreve o turno do evento recorrente; ele não demonstra, por si só, que o alarme anterior pertence ao mesmo turno, distinção que deve ser preservada na interpretação.

Esses achados sustentam o uso do IAT como base de uma *proxy* quantitativa de regimes em cascata, mas não comprovam propagação causal entre componentes. Para a modelagem, a intensidade diária do fenômeno será definida pelo número de eventos cujo IAT seja positivo e inferior a 30 minutos, atribuindo-se cada recorrência ao dia do alarme subsequente.

## 4.5 Análise Espectral via Transformada de Fourier

A análise espectral foi aplicada à série diária para identificar periodicidades que não são facilmente separáveis por inspeção visual. Antes da Transformada Rápida de Fourier (FFT), a série foi reindexada em frequência diária, os dias de contagem zero foram interpolados apenas para a estimação espectral e removeu-se a tendência linear. A potência das frequências positivas foi então convertida em função do período, em dias, e agrupada em faixas interpretáveis. O procedimento foi repetido na série bruta e na série sem outliers.

Na série bruta, a faixa de 2 a 4 dias concentrou 28,50% da potência espectral, seguida pela faixa semanal, de 5 a 9 dias, com 17,38%. As faixas quinzenal e mensal responderam por 7,05% e 7,06%, respectivamente. Componentes de 36 a 90 dias concentraram 8,65%, enquanto as faixas trimestral, semestral, anual e plurianual reuniram 6,07%, 3,65%, 7,22% e 7,70%. Os picos de maior potência dentro das principais faixas ocorreram em aproximadamente 2,34, 7,97, 16,14 e 30,05 dias.

Após a remoção dos outliers, a participação da faixa de 2 a 4 dias diminuiu para 22,55% e a semanal para 11,78%. Essa redução mostra que os valores extremos amplificavam as oscilações rápidas, mas não as criavam, pois ambas permanecem relevantes na série tratada. Em contrapartida, aumentaram as participações relativas das faixas mensal (8,24%), bimestral (11,96%), trimestral (8,16%) e plurianual (12,80%). Como os percentuais são normalizados pela potência total, esses aumentos não significam necessariamente crescimento absoluto das componentes longas; parte do efeito decorre da perda de potência nas frequências curtas.

As participações quinzenal e anual permaneceram praticamente constantes, em torno de 7%. Entretanto, somente a faixa anual preservou também o mesmo pico estimado, em 325,5 dias; o pico da faixa quinzenal deslocou-se de 16,14 para 10,17 dias. Os ciclos longos devem ser interpretados com cautela, pois a série contém poucos ciclos anuais completos, tendência estrutural, anos parciais e hiatos extensos. Em conjunto com as análises de sazonalidade e autocorrelação, a FFT oferece suporte mais robusto à persistência de padrões de curto prazo e de uma componente semanal.

## 4.6 Decomposição Aditiva da Série Temporal

A série diária também foi decomposta de forma aditiva em tendência, componente semanal, componente mensal e resíduo. A implementação emprega uma média móvel centrada de 30 dias para estimar a tendência; sobre a série sem tendência, calcula-se o efeito médio de cada dia da semana; e, após sua remoção, estima-se o efeito médio de cada mês. O resíduo corresponde à parcela não explicada por essas três componentes. O método é uma decomposição aditiva sequencial própria e não deve ser confundido com o algoritmo STL.

Na série bruta, a amplitude da tendência foi de 219,83 alarmes, a amplitude semanal de 13,36, a mensal de 6,60 e o desvio-padrão residual de 50,67. A presença de picos superiores a 600 e 800 alarmes por dia elevou a tendência local e concentrou grande parte da variação não explicada no resíduo. Ainda assim, a tendência revela mudanças persistentes: níveis mais elevados em partes de 2020 e 2021, oscilações durante 2022, forte redução em 2023 e recuperação a partir de 2024. Esses padrões descrevem o comportamento do registro de alarmes e não devem ser convertidos diretamente em afirmações sobre estabilidade da produção, para a qual não há medida de exposição disponível.

Na série sem outliers, a amplitude da tendência caiu para 100,21 alarmes, a semanal para 1,50, a mensal para 3,92 e o desvio-padrão residual para 24,72. A redução da amplitude semanal foi particularmente expressiva, confirmando que os valores extremos amplificavam o contraste entre dias da semana. A componente mensal apresentou maior estabilidade relativa, enquanto a forma geral da tendência foi preservada. Isso indica que as mudanças de longo prazo não eram produzidas exclusivamente pelos outliers, embora sua magnitude fosse substancialmente inflada por eles.

Os trechos de zero alarme permanecem visíveis como observações originais nos gráficos, mas são interpolados na série de trabalho utilizada para estimar as componentes. Assim, a tendência traçada dentro de um hiato representa uma continuidade estatística estimada, não atividade efetivamente observada. A decomposição é descritiva e utiliza informação de ambos os lados da janela centrada; suas componentes não devem ser inseridas diretamente como *features* preditivas sem uma reformulação causal.

## 4.7 Anomalias Cíclicas

A caracterização categórica mostrou que a distribuição de alarmes entre as 20 máquinas é relativamente dispersa. TR10 registrou o maior volume, com 6.379 eventos, seguida por TR17, com 5.527; ainda assim, nenhuma máquina domina individualmente o conjunto, e a análise de Pareto não revelou uma concentração comparável à regra clássica de 80/20. A diferença entre os extremos é relevante — TR16 possui 1.639 alarmes —, mas o fenômeno está distribuído por toda a linha. Entre os tratos, as categorias 26 e 18 se destacam com 15.962 e 13.160 eventos, volumes substancialmente superiores aos demais.

Os códigos de alarme são fortemente concentrados em zero: FC1 apresenta 78.155 registros iguais a zero, FC2 apresenta 85.134 e FC3 apresenta 45.888. FC1 e FC3 exibiram correlação numérica próxima de 1, enquanto FC2 apresentou associação próxima de zero com ambos. Essa correlação não implica igualdade literal — FC1 e FC3 coincidem em 51,90% das linhas — e deve ser interpretada com cautela, pois os códigos são categorias representadas por números. A ausência de um dicionário operacional e de informação de severidade impede inferir que códigos iguais a zero correspondam a alarmes inexplicados ou menos críticos. Por essa razão e pela ausência de relevância adicional identificada nas análises realizadas, os códigos foram descartados da engenharia de *features*, assim como os demais campos operacionais não selecionados.

A distribuição intradiária apresentou pico geral às 6h e maior concentração entre 6h e 16h. Esse comportamento coincide com o início do primeiro turno e a transição para o segundo, mas não permite concluir que o volume de produção seja máximo nesse intervalo, uma vez que a base não contém unidades produzidas nem horas efetivamente trabalhadas. O terceiro turno concentra menos eventos em termos absolutos. No mapa hora por dia da semana, o domingo apresenta elevações no início do primeiro turno e, com maior intensidade, na transição entre 15h e 16h.

Na série bruta, o domingo reuniu 15.339 alarmes, enquanto os demais dias variaram entre 11.845 e 12.617. Considerando apenas dias com atividade registrada, a média dominical foi de 73,00 alarmes, 15,14 acima da média global; nos outros dias, as médias ficaram entre 53,74 e 57,56. Após o tratamento dos outliers, as médias condicionais tornaram-se muito mais próximas: 46,98 aos domingos e entre 45,20 e 47,58 nos demais dias. Portanto, a anomalia dominical é real na série bruta, mas é produzida principalmente por episódios extremos e não constitui, isoladamente, uma sazonalidade semanal uniforme ao longo de todo o período.

A distribuição mensal bruta atingiu seu maior volume em junho e apresentou elevações adicionais em março, maio, julho e setembro. A análise sem outliers reduziu os contrastes, mas preservou uma componente mensal de menor amplitude, também observada na decomposição aditiva. Essas anomalias de calendário justificam o uso de variáveis temporais na modelagem, desde que sejam codificadas sem acesso a informações posteriores ao instante de previsão.

## 4.8 Autocorrelação da Série Diária

A função de autocorrelação (ACF) foi calculada até 90 dias nas séries bruta e sem outliers. Em ambas, os coeficientes são positivos e decaem lentamente, permanecendo acima do intervalo de confiança de 95% em praticamente todos os lags avaliados. Esse comportamento evidencia memória temporal, mas também reflete tendência, mudanças de regime e longas sequências de zeros; portanto, a significância isolada das barras não implica que um modelo deva incorporar todas as 90 defasagens.

Na série bruta, a autocorrelação parcial (PACF) identificou efeitos diretos em lags curtos e em pontos específicos, incluindo 7, 14, 21, 27, 28 e 29 dias. Na série sem outliers, os lags 1 a 5, 7 e 14 permaneceram entre os sinais mais consistentes, enquanto efeitos mais longos se tornaram esparsos. A remoção dos extremos elevou a ACF global de curto prazo e tornou seu decaimento mais regular, comportamento esperado porque a interpolação suaviza transições entre dias vizinhos. Esse aumento não deve ser interpretado automaticamente como fortalecimento físico da dependência.

A análise anual revelou heterogeneidade temporal. Em 2021, a série bruta apresentou sinais semanais nos lags 7, 14, 21 e 28; em 2024, os lags 28 e 30 tornaram-se levemente negativos; e, em 2025, a autocorrelação bruta de um dia alcançou 0,675, mas caiu para 0,368 após a remoção dos outliers. Esse último resultado sugere que os extremos de 2025 estavam temporalmente agrupados. Como 2025 contém somente dados até maio, suas estimativas possuem menor comparabilidade com as dos anos completos. A divergência entre os perfis anuais sugere estrutura temporal não estacionária, hipótese que deverá ser examinada formalmente durante a modelagem e considerada na validação cronológica.

## 4.9 Correlação Temporal com Variáveis Exógenas

A associação entre o volume mensal de alarmes e as variáveis macroeconômicas foi avaliada com o coeficiente de Spearman, adequado a relações monotônicas sem pressupor normalidade. As séries foram harmonizadas pela chave mensal *AnoMes*. Como a fonte do PIB contém meses duplicados, seus coeficientes devem ser interpretados como exploratórios e com cautela adicional; a variável foi descartada da engenharia de *features* final.

Na série bruta, a taxa Selic apresentou correlação de -0,489 com o volume mensal de alarmes (*p* = 0,00005), o PIB apresentou -0,359 (*p* = 0,00382), a balança comercial apresentou -0,119 (*p* = 0,35232) e o IPCA de bens duráveis apresentou 0,434 (*p* = 0,00038). Na série sem outliers, os coeficientes foram, respectivamente, -0,410 (*p* = 0,00085), -0,267 (*p* = 0,03438), -0,075 (*p* = 0,56048) e 0,401 (*p* = 0,00113). Selic e IPCA de bens duráveis preservaram direção, magnitude moderada e significância estatística nas duas versões; a balança comercial não apresentou evidência de associação; e o PIB, embora significativo, exibiu efeito mais fraco e sensível ao tratamento dos extremos.

Foram também avaliadas defasagens mensais entre -3 e +3 meses. Nenhuma variável apresentou ganho de associação suficientemente forte e estável que justificasse, apenas pela EDA, a seleção de múltiplos lags macroeconômicos. Esses resultados são associativos, não causais: tendências compartilhadas, paralisações e o pequeno número efetivo de meses podem gerar correlações espúrias. Para evitar vazamento temporal, qualquer variável exógena utilizada no modelo deverá ser associada à data em que se encontrava efetivamente disponível, e não simplesmente repetida em todos os dias do mês de referência.
# 5. Engenharia de *Features*

## 5.1 Estratégia de Construção e Visão Geral

A engenharia de *features* foi redesenhada a partir do alvo de recorrência definido na EDA. Cada linha das matrizes representa o dia a ser previsto, enquanto toda variável operacional utiliza exclusivamente informação disponível até o término do dia anterior. Essa convenção impede que o volume ou as recorrências do próprio dia sejam utilizados para prever um alvo construído a partir dos mesmos eventos. As variáveis de calendário constituem a única exceção temporal, pois são conhecidas antecipadamente.

O pipeline realiza uma única limpeza da base de eventos, calcula o IAT por máquina e constrói, no fuso horário local da planta, uma base diária intermediária com 1.953 datas. Nessa representação, 1.527 dias possuem ao menos um registro de alarme e 426 dias são marcados como potencial indisponibilidade. A diferença de um dia em relação à agregação exploratória em UTC — 1.528 dias com registros e 425 dias de contagem zero — decorre da conversão para `America/Recife` antes da atribuição de cada evento ao dia operacional. A aplicação do maior lag, de 28 dias, e a exclusão conservadora dos dias sem registros resultam em 1.499 observações completas para modelagem.

Foram produzidos dois conjuntos finais, alinhados pelas mesmas datas e pelos mesmos valores do alvo:

| Conjunto | Observações elegíveis | Preditores | Composição |
|---|---:|---:|---|
| Simples | 1.499 | 16 | Calendário, histórico de gaps e volume global de alarmes com defasagem |
| Completo | 1.499 | 36 | Todas as variáveis do conjunto simples, Selic, IPCA de duráveis e volumes defasados de TR10, TR17 e TR19 |

Ambos preservam a data do alvo, a contagem `recurrences_lt_30m` e o metadado `target_available`. As verificações finais confirmaram ordenação cronológica, unicidade das datas, alvo não negativo e ausência de valores ausentes entre os preditores das linhas elegíveis. Também foi verificado que o conjunto simples não contém variáveis específicas de máquina, enquanto o completo contém os seis lags previstos para cada uma das três máquinas selecionadas.

## 5.2 *Features* temporais e cíclicas

As variáveis de calendário são *Year*, *Month*, *Day*, *DayOfWeek*, *WeekOfYear* e *IsWeekend*. Elas descrevem a posição conhecida do dia-alvo no calendário e permitem que os modelos capturem mudanças graduais, padrões semanais, efeitos mensais e diferenças entre dias úteis e finais de semana. Embora a EDA tenha identificado anomalias em horários e transições de turno, a granularidade da modelagem é diária; por isso, a hora do evento e o turno não são empregados diretamente como atributos do dia.

## 5.3 *Features* de Janelas Deslizantes

Médias móveis de três e sete dias foram consideradas em uma versão preliminar do pipeline, mas não foram mantidas nos conjuntos finais. Além de introduzirem redundância com os volumes defasados, janelas que incluam o próprio dia produziriam vazamento do alvo. A versão final prioriza lags pontuais, cuja origem temporal é explícita e auditável. Componentes da decomposição aditiva também não são utilizadas como *features*, pois foram estimadas com janela centrada e possuem finalidade exclusivamente descritiva na EDA.

## 5.4 *Features* de Volume com Defasagem

O volume diário global de alarmes foi defasado em 1, 3, 7, 14, 21 e 28 dias. Os lags de 1 e 3 dias representam a memória de curto prazo observada na ACF e na PACF; 7 e 14 dias capturam a estrutura semanal mais consistente; e 21 e 28 dias permitem avaliar persistências mensais aproximadas identificadas em anos específicos. Nenhum volume do próprio dia-alvo é incluído. O pipeline utiliza o volume bruto nos dias anteriores, preservando precisamente os episódios de alta atividade que podem anteceder recorrências, e não aplica interpolação futura ou limites calculados com observações posteriores ao corte de treino.

## 5.5 *Features* de Recorrências e Cascata

O IAT é calculado após a ordenação cronológica dos eventos dentro de cada máquina. Para um evento $i$ da máquina $m$, define-se $IAT_{i,m}$ como a diferença, em minutos, entre seu instante e o alarme imediatamente anterior de $m$. O primeiro registro de cada máquina não possui IAT e não é contabilizado. A base completa contém 88.391 intervalos positivos válidos.

Uma recorrência rápida ocorre quando $0 < IAT_{i,m} < 30$ minutos. A agregação pelo dia do evento subsequente produziu 31.712 recorrências no período completo. Essa quantidade é utilizada como alvo, não como preditor contemporâneo. O desenho atual também não inclui lags do próprio alvo, mantendo o modelo simples fundamentado em calendário, qualidade da continuidade e histórico de volume de alarmes. A eventual inclusão de recorrências passadas poderá ser avaliada posteriormente sem alterar a definição do alvo.

## 5.6 *Features* de Contexto de Hiatos

Dias com contagem zero não são assumidos automaticamente como operação normal sem recorrências. Na base intermediária, esses dias recebem `target_available = 0`; suas datas e volumes são preservados para manter a continuidade cronológica, mas eles são excluídos por padrão da matriz de treinamento. Trata-se de uma decisão conservadora diante da impossibilidade de separar, apenas pelos alarmes, ausência real de eventos, paralisação de planta e indisponibilidade de registros.

Quatro atributos causais transportam o contexto do gap para os dias seguintes: `GapFlag_lag1d`, indicador de que o dia anterior possui zero registros; `GapRunLength_lag1d`, comprimento da sequência de zeros encerrada ou vigente no dia anterior; e `GapDays_last7d` e `GapDays_last30d`, números de dias sem registros nas janelas históricas correspondentes. Assim, o primeiro dia com atividade após os hiatos de 70 e 90 dias carrega explicitamente comprimentos 70 e 90, sem consultar datas posteriores ao alvo.

## 5.7 *Features* de Máquinas

TR10 e TR17 foram selecionadas por apresentarem os maiores volumes globais de alarmes, enquanto TR19 foi selecionada por possuir o menor IAT mediano e a maior taxa de recorrência no limiar de 30 minutos. Para cada uma, o modelo completo incorpora o volume diário defasado em 1, 3, 7, 14, 21 e 28 dias. Os 18 atributos resultantes permitem representar concentrações localizadas de atividade sem usar o volume contemporâneo das máquinas, que seria parte direta da informação empregada na construção do alvo do próprio dia.

## 5.8 Variáveis Exógenas Macroeconômicas

O conjunto completo inclui exclusivamente Selic e IPCA de bens duráveis, variáveis que preservaram associações moderadas e estatisticamente significativas nas séries bruta e sem outliers. Para representar disponibilidade realista, todos os dias de um mês recebem o valor do último mês concluído, com atraso mínimo de um mês. Valores ausentes nas bordas não são preenchidos a partir do futuro.

O PIB foi descartado porque apresentou associação mais fraca, sensível ao tratamento dos outliers e sem ganho suficientemente claro para compensar sua baixa frequência temporal. A balança comercial foi descartada por apresentar correlações fracas e não significativas nas duas séries. Os lags macroeconômicos adicionais de -3 a +3 meses também não foram selecionados, pois a EDA não identificou melhora estável de associação. Essa seleção reduz a dimensionalidade e evita incorporar variáveis mensais com justificativa apenas circunstancial.

# 6. Modelagem Preditiva

## 6.1 Definição do Alvo de Predição e Horizonte Temporal

A variável-alvo é uma contagem diária não negativa. Para cada dia (d), define-se:

\[
y_d = \sum_i \mathbb{1}\left(\operatorname{data}(i)=d \;\land\; 0 < IAT_i < 30\right),
\]

em que $i$ representa um evento de alarme, $IAT_i$ é o intervalo desde o alarme anterior da mesma máquina e $\mathbb{1}$ é a função indicadora. O horizonte adotado é de um dia: o modelo estima $y_d$ utilizando calendário conhecido e informações operacionais disponíveis até $d-1$. A tarefa é, portanto, de regressão de contagem. O alvo quantifica recorrência rápida e funciona como *proxy* da intensidade de um regime de alarmes em cascata; não representa um rótulo externo de falha nem comprova propagação causal entre máquinas.

## 6.2 Estratégia de Validação

A avaliação foi dividida em duas etapas. Inicialmente, os últimos 365 dias de calendário foram isolados como teste final e permaneceram fora da seleção de algoritmos e hiperparâmetros. Esse período compreende 17 de maio de 2024 a 16 de maio de 2025 e contém 359 observações com alvo disponível. O período de desenvolvimento reúne 1.140 observações entre 8 de fevereiro de 2020 e 16 de maio de 2024.

Dentro do período de desenvolvimento, a validação utilizou janela expansiva com 730 dias iniciais de histórico, janelas nominais de validação de 180 dias e avanço de 180 dias. O procedimento produziu cinco folds. Em cada iteração, todo o histórico anterior foi incorporado ao treinamento e a validação ocorreu exclusivamente em datas posteriores. Como dias sem alvo disponível foram excluídos, o número de observações por janela variou entre 79 e 138 e, em alguns folds, a primeira data efetivamente avaliada foi posterior ao início nominal da janela.

Os dias marcados como indisponíveis e as primeiras linhas sem histórico completo de 28 dias não participam do ajuste nem da avaliação. Nenhum embaralhamento ou divisão aleatória é aplicado. Essa estratégia respeita a não estacionariedade identificada na ACF anual e mede a capacidade de generalização para regimes cronologicamente posteriores.

## 6.3 Modelos, Configurações Experimentais e *Baseline*

Foram avaliados dois algoritmos de regressão baseados em árvores. O *Random Forest Regressor* combina predições de múltiplas árvores construídas sobre amostras e subconjuntos de atributos, reduzindo a variância em relação a uma árvore individual. O *XGBoost Regressor* constrói árvores sequencialmente e foi configurado com o objetivo `count:poisson`, coerente com a natureza não negativa da variável-alvo. As predições foram limitadas ao domínio não negativo e nenhuma normalização foi aplicada, pois os dois algoritmos não dependem de escala uniforme entre os preditores.

Cada algoritmo foi treinado com os conjuntos simples e completo, formando quatro experimentos: Random Forest simples, Random Forest completo, XGBoost simples e XGBoost completo. Para cada combinação foram avaliadas seis configurações compactas de hiperparâmetros. A seleção utilizou exclusivamente o menor MAE médio nos cinco folds expansivos; o RMSE foi empregado como primeiro critério de desempate. Todos os ajustes utilizaram semente aleatória 42.

Como referência, foi implementado um *baseline* de persistência que utiliza a quantidade de recorrências observada no dia civil anterior. O cálculo só é possível quando o alvo desse dia anterior está disponível; por isso, sua cobertura no teste final foi de 357 das 359 observações, ou 99,4%. Essa diferença de cobertura foi mantida explícita na comparação.

As métricas selecionadas foram MAE, RMSE, coeficiente de determinação ($R^2$) e desvio médio de Poisson. O MAE foi adotado como critério principal por fornecer o erro médio diretamente em número de recorrências. O RMSE atribui peso maior aos erros extremos; o $R^2$ compara a capacidade explicativa com a predição pela média; e o desvio de Poisson complementa a avaliação da distribuição de contagens. O MAPE não foi utilizado porque o alvo contém valores zero.

## 6.4 Resultados da Validação Expansiva

As melhores configurações de cada experimento apresentaram os seguintes resultados médios no período de desenvolvimento:

| Algoritmo | Conjunto | MAE médio | Desvio-padrão do MAE | RMSE médio |
|---|---|---:|---:|---:|
| Random Forest | Simples | 17,000 | 4,905 | 27,925 |
| Random Forest | Completo | 16,707 | 5,403 | 27,422 |
| XGBoost | Simples | 13,943 | 4,849 | 26,848 |
| XGBoost | Completo | **13,821** | **4,683** | **26,412** |

O XGBoost completo obteve o menor MAE médio, mas sua vantagem sobre o XGBoost simples foi de apenas 0,122 recorrência, ou aproximadamente 0,9%. Além disso, o modelo completo superou o simples em somente dois dos cinco folds. As médias do alvo variaram de 1,76 a 20,77 recorrências entre as janelas, e o $R^2$ do XGBoost completo permaneceu negativo em todos os folds. Esses resultados confirmam que o comportamento temporal é não estacionário e que a vantagem das variáveis adicionais não se mantém com a mesma intensidade em todos os regimes históricos.

As duas melhores configurações da busca do XGBoost completo também ficaram muito próximas: MAE médio de 13,821 para a configuração selecionada e 13,885 para a segunda colocada. Assim, não se observa um ótimo isolado fortemente definido dentro do espaço avaliado. A configuração escolhida possui 300 árvores, taxa de aprendizado de 0,05, profundidade máxima 6, peso mínimo de filho 1, amostragem de linhas e colunas de 0,8 e regularização L2 igual a 1.

## 6.5 Desempenho no Teste Temporal Isolado

Após a seleção dos hiperparâmetros, cada estimador foi reajustado utilizando as 1.140 observações do período de desenvolvimento e avaliado uma única vez nas 359 observações do teste final. Os resultados são apresentados a seguir:

| Modelo | Conjunto | Observações | Cobertura | MAE | RMSE | $R^2$ | Desvio de Poisson |
|---|---|---:|---:|---:|---:|---:|---:|
| XGBoost | Completo | 359 | 100% | **8,817** | **15,048** | **0,051** | **12,762** |
| XGBoost | Simples | 359 | 100% | 9,267 | 15,302 | 0,019 | 13,148 |
| Random Forest | Simples | 359 | 100% | 10,994 | 16,746 | -0,175 | 14,639 |
| Random Forest | Completo | 359 | 100% | 11,155 | 16,337 | -0,118 | 14,460 |
| *Baseline* lag-1 | Alvo anterior | 357 | 99,4% | 11,462 | 20,425 | -0,743 | 62,747 |

O XGBoost completo apresentou o melhor valor em todas as métricas reportadas entre os modelos treinados. Em relação ao XGBoost simples, reduziu o MAE em 4,9%, o RMSE em 1,7% e o desvio de Poisson em 2,9%. Na comparação restrita às 357 datas em que o *baseline* estava disponível, o MAE do modelo vencedor foi 8,828, correspondente a uma redução de aproximadamente 23,0% sobre a persistência diária. As duas configurações de Random Forest apresentaram $R^2$ negativo e tendência de superestimação da média: previram 13,85 e 14,39 recorrências por dia, respectivamente, diante de média observada de 10,80.

Apesar da primeira posição, o $R^2$ de 0,051 indica que o XGBoost completo explica apenas uma pequena parcela da variabilidade do período final. Sua superioridade deve ser interpretada como o melhor desempenho entre as alternativas avaliadas, e não como evidência de elevada precisão preditiva.

## 6.6 Importância das *Features* no Modelo Vencedor

A importância normalizada exposta pelo XGBoost completo foi agregada por grupo de variáveis:

| Grupo de *features* | Importância agregada |
|---|---:|
| Volumes defasados de TR10, TR17 e TR19 | 48,8% |
| Indicadores de gaps | 22,1% |
| Volume global de alarmes com defasagem | 19,0% |
| Variáveis de calendário | 5,8% |
| Selic e IPCA de duráveis | 4,3% |

Os volumes específicos das três máquinas concentraram quase metade da importância, com destaque para `TR19_AlarmVolume_lag1d`, segunda variável individual mais relevante, com 8,0%. A primeira foi `GapFlag_lag1d`, com 8,9%, seguida por `GapDays_last30d`, `AlarmVolume_lag1d`, `AlarmVolume_lag14d`, `GapDays_last7d` e `TR10_AlarmVolume_lag1d`. A Selic respondeu por 2,7% e o IPCA de duráveis por 1,5%, reforçando que as variáveis macroeconômicas desempenharam papel secundário.

As variáveis acrescentadas pelo conjunto completo somaram 53,1% da importância interna. Entretanto, essa proporção não deve ser interpretada como efeito causal nem como ganho preditivo de mesma magnitude: a redução do MAE em relação ao modelo simples foi de 4,9% no teste e inferior a 1% na média da validação expansiva. Parte da importância pode representar substituição ou redundância em relação ao volume global de alarmes.

## 6.7 Análise dos Erros e Limitações Preditivas

No teste final, o alvo apresentou média de 10,80 e mediana de 6 recorrências, enquanto a média prevista foi de 10,04. A diferença média de 0,76 indica boa calibração agregada, com leve subestimação. Essa proximidade entre as médias, contudo, oculta uma forte compressão da amplitude: os valores observados variaram de 0 a 131, ao passo que as previsões ficaram entre 1,53 e 23,65. A correlação linear entre observado e previsto foi de apenas 0,236 e o erro absoluto mediano foi de 5,62 recorrências.

O modelo superestimou 237 dos 359 dias, principalmente aqueles com baixa atividade, e subestimou 122. Nos 32 dias com alvo zero, previu em média 7,44 recorrências. Por outro lado, os erros positivos pouco frequentes foram muito maiores: nos 38 dias com alvo entre 21 e 50, o MAE foi 18,11; nos 11 dias acima de 50, a média observada foi 78,45, a previsão média foi 14,33 e o MAE alcançou 64,12. No maior pico, de 131 recorrências em 8 de setembro de 2024, a previsão foi 9,93.

A comparação pareada com o XGBoost simples mostra que o completo foi melhor em 180 dias e pior em 179. O ganho concentrou-se nas faixas mais frequentes entre 1 e 20 recorrências; para alvos acima de 20, o conjunto completo apresentou erro médio ligeiramente maior. Portanto, as novas variáveis melhoraram o desempenho global sobretudo em dias de atividade baixa a moderada, sem resolver a antecipação de eventos extremos.

Os resíduos ainda apresentaram correlação de 0,224 na defasagem exata de sete dias e de 0,160 em quatorze dias, indicando que parte da estrutura semanal permaneceu sem representação adequada. Em conjunto com a variação acentuada entre folds, esse resultado sugere que a principal limitação não é apenas algorítmica, mas também informacional: calendário, histórico de volumes, gaps, máquinas selecionadas e variáveis macroeconômicas não são suficientes para caracterizar integralmente as mudanças de regime e os picos raros de recorrência.

Conclui-se que o XGBoost completo constitui o modelo final desta pesquisa por apresentar o menor MAE no teste temporal isolado e superar o *baseline* e as demais combinações avaliadas. Seu uso deve, contudo, ser entendido como uma estimativa da intensidade cotidiana de recorrências, com capacidade limitada para antecipar picos extremos e com necessidade de recalibração diante de mudanças no regime operacional.
# 7. Discussão

## 7.1 Contribuições Científicas

A principal contribuição científica deste trabalho decorre da utilização de dados reais produzidos por uma planta automotiva em operação. Revisões da área de manutenção preditiva apontam que grande parte dos estudos é desenvolvida com dados simulados, bases laboratoriais ou conjuntos públicos já estruturados, enquanto aplicações documentadas com dados fabris reais permanecem menos frequentes **[Zonta et al., 2020]**. Restrições de confidencialidade, heterogeneidade dos sistemas industriais, ausência de padronização e dificuldade de validação operacional contribuem para essa lacuna. A pesquisa demonstra, portanto, como registros originados diretamente de um sistema industrial podem ser transformados em uma base analítica reproduzível, preservando limitações e descontinuidades que seriam menos evidentes em um conjunto previamente curado.

O estudo também se insere em um tema com literatura aplicada ainda restrita: os regimes de alarmes em cascata e seu impacto sobre a operação industrial. Trabalhos sobre manutenção preditiva frequentemente se concentram em sensores, classificação de falhas ou estimação de vida útil, enquanto há menos estudos dedicados à concentração temporal de alarmes e à antecipação de períodos nos quais recorrências rápidas dificultam a atuação dos operadores. A comparação direta com trabalhos semelhantes é ainda limitada pela escassez de bases públicas que preservem a ordem dos eventos, a identidade das máquinas e o contexto operacional necessário para caracterizar recorrências.

Diante da ausência de um rótulo externo de cascata, esta pesquisa propôs uma definição operacional e reprodutível baseada no tempo entre alarmes da mesma máquina. A contagem diária de eventos com IAT positivo inferior a 30 minutos não comprova propagação causal entre equipamentos, mas fornece uma medida objetiva da intensidade de recorrências rápidas. Essa definição permitiu converter um fenômeno operacional pouco estruturado em um problema supervisionado de regressão de contagem, tornando possível avaliar quantitativamente diferentes conjuntos de atributos e algoritmos.

Outra contribuição é a documentação integrada do percurso entre dado bruto e avaliação final. O trabalho compreende a identificação de hiatos, comparação entre séries bruta e sem outliers, análise de recorrência por máquina e turno, avaliação espectral, decomposição, autocorrelação, seleção de variáveis exógenas, construção de atributos e validação temporal com teste final isolado. A explicitação dos dias com possível indisponibilidade e a manutenção de indicadores de qualidade impedem que todo zero seja interpretado automaticamente como operação normal. O código modular e os *notebooks* executáveis tornam esse percurso auditável e adaptável a outros conjuntos de registros de alarmes.

## 7.2 Implicações Práticas para a Indústria

Embora o XGBoost completo tenha obtido o menor erro no teste final, o modelo simples apresenta uma relação particularmente favorável entre desempenho e custo de implantação. Sua versão com XGBoost alcançou MAE de 9,267, apenas 0,451 recorrência acima do modelo completo, utilizando 16 preditores em vez de 36. O conjunto simples depende exclusivamente do calendário, do histórico do volume global de alarmes e dos indicadores de gaps; não exige atualização de séries macroeconômicas nem manutenção de agregações específicas para TR10, TR17 e TR19. Essa autonomia reduz dependências de integração e facilita sua incorporação à infraestrutura já utilizada para armazenar os alarmes da planta parceira.

Na prática, o *pipeline* pode ser executado diariamente após o encerramento do dia operacional. A partir dos registros já coletados, calculam-se os volumes históricos e os indicadores de continuidade necessários para estimar a quantidade de recorrências rápidas esperada no dia seguinte. A previsão não identifica uma falha específica nem determina sua causa, mas pode funcionar como sinal antecipado de aumento da pressão operacional sobre as equipes responsáveis pelo acompanhamento dos alarmes.

Esse sinal pode apoiar o planejamento diário da manutenção preditiva. Dias com previsão elevada podem motivar reforço do monitoramento, revisão das ocorrências recentes, priorização de inspeções preventivas e adequação da disponibilidade de técnicos e recursos. A previsão também pode ser incorporada a um painel operacional, acompanhada dos indicadores de gaps, para que o usuário saiba quando o histórico recente foi afetado por interrupções ou indisponibilidade de dados. Os limiares de alerta devem ser definidos em conjunto com especialistas da planta, considerando a capacidade da equipe e o custo relativo de falsos alertas e de picos não antecipados.

O modelo simples não substitui o diagnóstico de engenharia nem deve acionar intervenções automaticamente. Seu baixo $R^2$ e a subestimação dos maiores picos indicam que a previsão é mais adequada como informação complementar para priorização do que como estimativa precisa da carga máxima. Uma implementação piloto deve manter a decisão com os profissionais responsáveis, registrar previsões e eventos observados e acompanhar degradação de desempenho ao longo do tempo. A atualização periódica do modelo é necessária porque a análise temporal mostrou mudanças relevantes entre os regimes históricos da planta.

## 7.3 Limitações e Ameaças à Validade

A primeira limitação é a ausência de classificação de severidade dos alarmes. Todos os registros possuem a mesma representação na base, independentemente de exigirem intervenção imediata, serem resolvidos automaticamente ou não produzirem impacto mensurável na linha. Consequentemente, o alvo atribui o mesmo peso a recorrências operacionalmente distintas. Um dia com grande número de alarmes pouco críticos pode receber valor superior ao de um dia com poucas ocorrências severas, restringindo a interpretação da contagem como indicador direto de risco, custo ou necessidade de manutenção.

Também não existe um rótulo validado de regime de cascata. O limiar de IAT inferior a 30 minutos foi estabelecido a partir da distribuição observada e oferece uma definição consistente, mas permanece como *proxy*. A base não permite demonstrar que um alarme causou outro, que ocorrências em máquinas diferentes pertencem ao mesmo processo de propagação ou que a concentração registrada produziu impacto sobre a produção. A validação dessa definição com especialistas da planta constitui condição importante para ampliar sua validade operacional.

A capacidade de generalização externa não foi testada. Os dados pertencem a uma única planta, a duas linhas da área *Body-in-White* e a um intervalo histórico específico. O desempenho em outras plantas, processos produtivos, famílias de equipamentos ou períodos posteriores a maio de 2025 permanece desconhecido. Perfis de máquina, políticas de manutenção, calendários, turnos e práticas de registro podem alterar tanto a distribuição do alvo quanto a relevância das *features*. Por isso, os resultados não devem ser transferidos diretamente para outro ambiente sem reconstrução do *pipeline*, recalibração e nova validação temporal.

A validade temporal dentro da própria planta também é limitada pela não estacionariedade. As médias do alvo variaram intensamente entre os folds expansivos, e o modelo vencedor apresentou $R^2$ negativo em todas as janelas de validação, apesar do resultado ligeiramente positivo no teste final. Os hiatos de origem parcialmente desconhecida reduzem a quantidade de observações elegíveis e podem coincidir com alterações operacionais não registradas. Embora os indicadores de gaps preservem esse contexto, eles não recuperam a informação ausente nem determinam a causa da interrupção.

Por fim, a capacidade preditiva permanece limitada para eventos extremos. O XGBoost completo comprimiu as previsões em uma faixa muito inferior à amplitude observada e subestimou os dias de maior recorrência. A importância interna das árvores indica quais variáveis foram utilizadas nas divisões do modelo, mas não estabelece causalidade e pode distribuir relevância entre atributos correlacionados. A busca de hiperparâmetros foi deliberadamente compacta e a comparação ficou restrita a Random Forest e XGBoost, decisão compatível com o tempo disponível, mas que não esgota as abordagens possíveis para a série.

# 8. Conclusões e Trabalhos Futuros

## 8.1 Conclusões

O objetivo geral da pesquisa foi desenvolver e documentar uma metodologia de análise exploratória, engenharia de *features* e modelagem preditiva aplicada a registros reais de alarmes industriais. Esse objetivo foi satisfeito. Os 88.411 eventos tratados foram organizados em uma série temporal contínua, investigados em suas dimensões temporal e operacional e convertidos em dois conjuntos de atributos para a previsão da quantidade diária de recorrências com IAT inferior a 30 minutos. A metodologia foi implementada em módulos reutilizáveis e exposta em *notebooks*, permitindo reproduzir o percurso entre dados brutos, análise e avaliação final.

O cumprimento dos objetivos específicos pode ser sintetizado da seguinte forma:

1. **Caracterização do conjunto de alarmes — satisfeito.** A estrutura, o volume, o período de coleta, a distribuição entre linhas e máquinas e as principais limitações da base foram documentados. A análise confirmou a natureza discreta dos eventos, a ausência de severidade e a impossibilidade de inferir diretamente o impacto operacional de cada registro.
2. **Diagnóstico e classificação dos hiatos — parcialmente satisfeito.** Foram identificadas 48 sequências de dias com zero alarmes, das quais 32 possuem ao menos dois dias, e períodos conhecidos foram associados à pandemia, paradas de planta e recessos de calendário. Entretanto, os hiatos de 70 e 90 dias permaneceram sem causa validada, impedindo a classificação completa apenas com os dados disponíveis.
3. **Análise de recorrência por máquina e turno — satisfeito.** O IAT foi calculado dentro de cada máquina e permitiu identificar TR10 e TR17 pelo volume elevado e TR19 pela maior concentração de recorrências rápidas. As transições de turno e o intervalo entre o segundo e o terceiro turnos apresentaram comportamento temporal distinto.
4. **Caracterização das sazonalidades e anomalias cíclicas — satisfeito.** A FFT, a decomposição aditiva e as comparações por calendário foram realizadas nas séries bruta e tratada. Os resultados mostraram componentes temporais de múltiplas escalas, mas também evidenciaram que parte do padrão dominical e de outras aparentes sazonalidades é amplificada por episódios extremos e mudanças de regime.
5. **Avaliação das variáveis exógenas — satisfeito.** Selic e IPCA de bens duráveis apresentaram as associações mais consistentes e foram incorporados ao conjunto completo com disponibilidade mensal conservadora. PIB e balança comercial foram descartados por associações fracas ou instáveis.
6. **Análise de autocorrelação — satisfeito.** Foram examinadas ACF e PACF das séries bruta e sem outliers, além da estabilidade anual dos coeficientes. A persistência de curto prazo e sinais semanais justificaram a escolha dos lags, enquanto a divergência entre anos reforçou a necessidade de validação temporal.
7. **Construção das *features* — satisfeito.** Foram obtidas matrizes simples e completa com 1.499 dias elegíveis e, respectivamente, 16 e 36 preditores. O desenho incorporou calendário, volumes defasados, histórico de gaps, variáveis por máquina e, no conjunto completo, Selic e IPCA de duráveis, mantendo os dias sem registros sob tratamento explícito de disponibilidade.
8. **Desenvolvimento e comparação dos modelos — satisfeito com limitações de desempenho.** Random Forest e XGBoost foram avaliados nos dois conjuntos de atributos por validação expansiva e teste temporal isolado. O XGBoost completo obteve o melhor resultado, com MAE de 8,817, RMSE de 15,048 e $R^2$ de 0,051. A análise de importância destacou os volumes das máquinas selecionadas e o contexto dos gaps, mas o baixo poder explicativo e a subestimação dos picos impedem interpretar o modelo como solução preditiva definitiva.

Em síntese, o trabalho demonstrou que é possível extrair sinal preditivo de registros reais de alarmes utilizando apenas informações disponíveis antes do dia-alvo. O conjunto completo ofereceu o melhor desempenho estatístico, enquanto o conjunto simples preservou resultado próximo com menor custo de integração. A principal conclusão não é que os regimes de recorrência tenham se tornado plenamente previsíveis, mas que o histórico de alarmes, quando tratado com atenção às descontinuidades e à ordem temporal, fornece uma base mensurável para apoiar o monitoramento e orientar pesquisas posteriores.

## 8.2 Proposta de Trabalhos Futuros

Uma primeira extensão consiste em avaliar modelos de rede neural recorrente, especialmente LSTM, capazes de receber sequências temporais em vez de vetores de lags previamente definidos. A comparação deverá utilizar os mesmos períodos de desenvolvimento, folds expansivos e teste final, evitando que diferenças na divisão temporal favoreçam uma arquitetura. O uso de redes neurais também exigirá controle de complexidade, regularização e quantidade de dados suficiente para verificar se a maior flexibilidade produz ganho real sobre os modelos de árvores.

A generalização das *features* e dos estimadores deve ser examinada com outros conjuntos de dados industriais semelhantes. Bases provenientes de outras linhas, plantas ou períodos posteriores permitiriam avaliar se os lags globais, os indicadores de gaps e a seleção de máquinas representam fenômenos transferíveis ou apenas características locais. A validação externa deverá contemplar recalibração do limiar de recorrência, comparação das distribuições e acompanhamento da degradação de desempenho.

Outra linha prioritária é aprofundar o estudo das variáveis nativas que não foram utilizadas nos modelos finais: *fault codes*, posições, *bilancellas* e tratos. Esses campos podem distinguir famílias de alarmes, regiões da linha e contextos operacionais associados a recorrências, mas sua inclusão depende de interpretação semântica, tratamento de alta cardinalidade e validação com especialistas da planta. Estudos de ablação devem medir o ganho incremental de cada grupo antes de ampliar o modelo implantado.

Também se recomenda construir uma classificação de severidade dos alarmes e obter rótulos operacionais associados a intervenções, paradas ou perda de produção. Com essas informações, o alvo poderá ponderar recorrências por criticidade e aproximar-se do impacto real sobre a manutenção. A definição de cascata baseada em IAT poderá então ser confrontada com avaliações de especialistas e com relações entre máquinas, permitindo distinguir mera concentração temporal de propagação operacional.

Por fim, modelos específicos para a assimetria e os eventos extremos devem ser investigados. Abordagens em duas etapas, com separação entre ocorrência de baixa atividade e previsão da intensidade positiva, funções de perda ponderadas ou modelos orientados a quantis podem reduzir a tendência de regressão à média observada neste trabalho. Paralelamente, um piloto prospectivo do modelo simples na planta parceira permitiria registrar previsões diariamente, definir limiares de alerta com a equipe de manutenção e avaliar se o sinal produzido contribui efetivamente para decisões operacionais.
# Referências
