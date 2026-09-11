---
tags:
  - IC
  - UNIFESP
data: 2026-09-10
---
# Revisão dos Novos Experimentos
---
## Justificativa e notas
A literatura mais abaixo foi escolhida pois contribui para justificar os experimentos adicionados à pesquisa.  O MLP avalia as relações não lineares entre os atributos construídos, enquanto LSTM avaliará a importância da sequencialidade temporal dos atributos, além da importância de *features* como os *lags* e classificadores temporais. A comparação final desses dois experimentos será feita a partir do resultado final dos *pipelines* completos — LSTM + Grid Search e MLP + Optuna — porque o método de otimização gera efeitos sobre os resultados.

**Nota sobre escrita do Paper:** Na escrita do paper, não deve ser mencionado que essas são partes novas adicionadas à pesquisa original. Tudo deve ser tratado como uma pesquisa única. O paper vai além do escopo de pesquisa da IC e é uma unidade a parte. Haverão agradecimentos à instituição da bolsa na seção de Acknowledgements, mas não se deve tratar como se houvesse uma pesquisa anterior que foi complementada. Todo o trabalho definido a partir deste ponto é um só.
## Síntese Teórica
### MLP
O _Multilayer Perceptron_ é uma rede _feedforward_ composta por densas camadas e transformações não lineares. Normalmente, os pesos são estimados por retropropagação do erro, procedimento consolidado por Rumelhart, Hinton e Williams (1986). Basta apenas uma camada oculta e adequação das funções de ativação para que a rede possua capacidade de aproximação universal; isso significa capacidade representacional, o que não é garantia de generalização nem determinação de quantos neurônios são necessários (Hornik, Stinchcombe e White, 1989).
### LSTM
Redes recorrentes mantêm um estado e o atualizam ao longo da sequência definida. No entanto, RNNs convencionais evidenciam dificuldade para aprender dependências distantes por causa do esvanescimento ou crescimento instável dos gradientes (Bengio, Simard e Frasconi, 1994). A LSTM foi proposta para preservar o fluxo de informação por intervalos maiores, utilizando estado de memória e mecanismos de controle aprendidos (Hochreiter e Schmidhuber, 1997).

No experimento proposto, a LSTM receberá janelas com um formato do tipo `amostras × passos temporais × atributos`. Este formato será utilizado para capitalizar a vantagem temporal da arquitetura.

**Nota sobre o texto do relatório da IC:** a formulação original de 1997 não continha exatamente todas as portas usadas na LSTM moderna. A **porta de esquecimento** foi introduzida posteriormente por Gers, Schmidhuber e Cummins (2000). Portanto, ao descrever entrada, saída e esquecimento em conjunto, convém citar ambos os trabalhos.
### Grid Search e Optuna
O Grid Search avalia o produto cartesiano de uma grade discreta de hiperparâmetros. É transparente e reproduzível, mas o custo computacional cresce multiplicativamente. Bergstra e Bengio (2012) demonstraram que a busca aleatória tende a cobrir melhor espaços nos quais apenas alguns hiperparâmetros são realmente importantes. Em virtude destes pontos, será utilizada uma grade pequena orientada para a LSTM, para que o custo computacional não seja determinante na avaliação do experimento proposto.

Optuna é um *framework* de otimização que permite espaços condicionais, amostradores sequenciais e interrupção antecipada de tentativas pouco promissoras (Akiba et al., 2019). Seu amostrador TPE deriva do trabalho de Bergstra et al. (2011), que modela separadamente regiões de desempenho favorável e desfavorável para escolher configurações futuras.

Buscando padronizar as métricas de avaliação dos experimentos propostos, a metodologia deve utilizar:
- os mesmos períodos de desenvolvimento e teste;
- os mesmos folds temporais;
- normalização ajustada somente sobre o treinamento de cada *fold*;
- a mesma métrica de seleção, o MAE médio (como já foi utilizado com XGBoost e Random Forest);
- orçamento computacional documentado;
- múltiplas sementes ou repetições, em virtude de redes neurais serem estocásticas.

Será utilizada a janela expansiva implementada nos experimentos com XGBoost e Random Forest. Esta metodologia foi proposta em virtude de que avaliações com origens móveis ou expansivas reduzem a dependência de um único corte temporal e reproduzem melhor o cenário real de previsão (Tashman, 2000).
