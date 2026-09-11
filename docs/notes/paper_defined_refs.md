---
tags:
  - IC
  - UNIFESP
data: 2026-09-10
---
# Literatura Escolhida
---
## Referências já Existentes
A base central está em [references.bib (line 528)](/C:/Users/daviz/Stuff/UNIFESP/Iniciação Científica - Didier/project_workspace/docs/relatorio/latex_relatorio/references.bib:528), enquanto o desenho experimental atual aparece em [relatorio_ic.md (line 208)](/C:/Users/daviz/Stuff/UNIFESP/Iniciação Científica - Didier/project_workspace/docs/relatorio/relatorio_ic.md:208).

1. **Hochreiter, S.; Schmidhuber, J. (1997). “Long Short-Term Memory.”**  
    Referência fundadora da LSTM. Deve sustentar a definição da arquitetura, sua memória interna e sua motivação como resposta às dificuldades das RNNs convencionais. [Neural Computation, 9(8), 1735–1780](https://doi.org/10.1162/neco.1997.9.8.1735).
    
2. **LeCun, Y.; Bengio, Y.; Hinton, G. (2015). “Deep Learning.”**  
    Revisão ampla apropriada para contextualizar redes profundas, aprendizado de representações e treinamento por gradiente. É útil como referência geral, mas não substitui trabalhos específicos sobre MLP ou LSTM. [Nature, 521, 436–444](https://doi.org/10.1038/nature14539).
    
3. **Bengio, Y. (2012). “Practical Recommendations for Gradient-Based Training of Deep Architectures.”**  
    Discute inicialização, otimização, seleção de hiperparâmetros e dificuldades práticas de treinamento. Sustenta decisões metodológicas como normalização, regularização e monitoramento da validação.
    
4. **Bergstra, J.; Bengio, Y. (2012). “Random Search for Hyper-Parameter Optimization.”**  
    Referência crítica para explicar limitações do Grid Search. Mostra que grades desperdiçam avaliações quando poucos hiperparâmetros dominam o desempenho. [JMLR, 13, 281–305](https://www.jmlr.org/papers/v13/bergstra12a.html).
    
5. **Kashpruk, N.; Piskor-Ignatowicz, C.; Baranowski, J. (2023). “Time Series Prediction in Industry 4.0.”**  
    Revisão do emprego de previsão temporal em manutenção, produção e detecção de anomalias. Fornece a ponte entre teoria de séries e aplicação industrial. [Applied Sciences, 13, 12374](https://doi.org/10.3390/app132212374).
    
6. **Zhao, R. et al. (2019). “Deep Learning and Its Applications to Machine Health Monitoring.”**  
    Organiza aplicações industriais de MLPs, autoencoders, CNNs e RNNs e discute a passagem de atributos manuais para aprendizado de representações. É uma referência central para posicionar os dois novos modelos. [Mechanical Systems and Signal Processing, 115, 213–237](https://doi.org/10.1016/j.ymssp.2018.05.050).
    
7. **Zonta, T. et al. (2020). “Predictive Maintenance in the Industry 4.0: A Systematic Literature Review.”**  
    Contextualiza modelos, infraestrutura e desafios da manutenção preditiva, destacando as dificuldades de aplicação em tempo real e em ambientes industriais reais. [Computers & Industrial Engineering, 150, 106889](https://doi.org/10.1016/j.cie.2020.106889).
    
8. **Smrekar, J. et al. (2010). “Prediction of Power Output of a Coal-Fired Power Plant by Artificial Neural Network.”**  
    Exemplo de rede neural aplicada como regressora sobre dados reais de uma planta. É comparável ao MLP proposto por utilizar atributos operacionais para estimar uma variável contínua, embora não trate alarmes nem recorrência temporal. [Neural Computing and Applications, 19, 725–740](https://doi.org/10.1007/s00521-009-0331-6).
    
**Nota sobre registro denominado `wang2020rul`** na base cuja combinação de título, autores e periódico não consegui corroborar. Antes de utilizá-lo, recomendo conferir seus metadados: o trabalho amplamente indexado com título semelhante é de Wang, Zhao e Addepalli, publicado na _Procedia Manufacturing_ ([DOI 10.1016/j.promfg.2020.06.015](https://doi.org/10.1016/j.promfg.2020.06.015)).
## Novas Referências
### Fundamentos de MLP e LSTM

1. **Rumelhart, D. E.; Hinton, G. E.; Williams, R. J. (1986). “Learning Representations by Back-Propagating Errors.”**  
    Trabalho clássico sobre ajuste dos pesos de redes multicamadas pela propagação reversa do erro. Deve ser citado na explicação do treinamento tanto do MLP quanto da LSTM. [Nature, 323, 533–536](https://doi.org/10.1038/323533a0).
    
2. **Hornik, K.; Stinchcombe, M.; White, H. (1989). “Multilayer Feedforward Networks Are Universal Approximators.”**  
    Fundamentação matemática da capacidade de aproximação de MLPs. Deve ser acompanhada da ressalva de que capacidade de aproximação não implica bom desempenho com amostras pequenas. [Neural Networks, 2(5), 359–366](https://doi.org/10.1016/0893-6080\(89\)90020-8).
    
3. **Bengio, Y.; Simard, P.; Frasconi, P. (1994). “Learning Long-Term Dependencies with Gradient Descent Is Difficult.”**  
    Demonstra teoricamente por que RNNs tradicionais têm dificuldade com dependências longas. É a principal referência para construir a motivação que conduz à LSTM. [IEEE Transactions on Neural Networks, 5(2), 157–166](https://doi.org/10.1109/72.279181).
    
4. **Gers, F. A.; Schmidhuber, J.; Cummins, F. (2000). “Learning to Forget: Continual Prediction with LSTM.”**  
    Introduz a porta de esquecimento, permitindo que a célula libere estados antigos em fluxos contínuos. É particularmente pertinente para dados industriais não segmentados naturalmente. [Neural Computation, 12(10), 2451–2471](https://doi.org/10.1162/089976600300015015).
    
5. **Zhang, G.; Patuwo, B. E.; Hu, M. Y. (1998). “Forecasting with Artificial Neural Networks: The State of the Art.”**  
    Revisão clássica de redes _feedforward_ para previsão. Discute escolha de entradas, número de camadas, neurônios, ativações e forte sensibilidade do resultado ao desenho experimental. [International Journal of Forecasting, 14(1), 35–62](https://doi.org/10.1016/S0169-2070\(97\)00044-7).
    
### Otimização e Validação

6. **Akiba, T. et al. (2019). “Optuna: A Next-Generation Hyperparameter Optimization Framework.”**  
    Referência oficial do Optuna. Apresenta espaços definidos dinamicamente, amostragem eficiente, paralelização e _pruning_ de tentativas. [KDD 2019, 2623–2631](https://doi.org/10.1145/3292500.3330701).
    
7. **Bergstra, J. et al. (2011). “Algorithms for Hyper-Parameter Optimization.”**  
    Fundamenta o TPE, abordagem sequencial apropriada para hiperparâmetros contínuos, discretos e condicionais. Sustenta a escolha do amostrador para o MLP. [NeurIPS 2011](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization).
    9. **Tashman, L. J. (2000). “Out-of-Sample Tests of Forecasting Accuracy: An Analysis and Review.”**  
    Fundamenta avaliações por origem móvel, recalibração e múltiplos períodos de teste. Dá suporte direto à manutenção dos _folds_ expansivos existentes. [International Journal of Forecasting, 16(4), 437–450](https://doi.org/10.1016/S0169-2070\(00\)00065-0).
    
8. **Srivastava, N. et al. (2014). “Dropout: A Simple Way to Prevent Neural Networks from Overfitting.”**  
    Justifica o _dropout_ como hiperparâmetro de regularização. É especialmente relevante diante das apenas 1.499 observações elegíveis do projeto. [JMLR, 15, 1929–1958](https://www.jmlr.org/papers/v15/srivastava14a.html).
    
### Implementações Comparáveis

9. **Silva, M. J.; Pereira, C. E. (2020). “Towards the Use of LSTM-Based Neural Network for Industrial Alarm Systems.”**  
    É o trabalho mais próximo do problema atual: utiliza uma LSTM regressora para prever comportamento de falhas em um sistema de alarmes. Sua limitação para a comparação é usar um processo industrial simulado. [DOI 10.48011/asba.v2i1.1436](https://doi.org/10.48011/asba.v2i1.1436).
    
10. **Mateus, B. C. et al. (2021). “Anticipating Future Behavior of an Industrial Press Using LSTM Networks.”**  
    Usa três anos de medições reais de uma prensa industrial e diferentes configurações LSTM para prever variáveis futuras. É útil para justificar janelas temporais, normalização e otimização da arquitetura. [Applied Sciences, 11(13), 6101](https://doi.org/10.3390/app11136101).
    
11. 15. **Taşcı, B.; Omar, A.; Ayvaz, S. (2023). “Remaining Useful Lifetime Prediction for Predictive Maintenance in Manufacturing.”**  
    Compara MLP, Random Forest, XGBoost e SVR sobre dados reais de sensores de linhas de montagem. Embora estime vida útil restante, oferece um paralelo forte com sua comparação entre redes e modelos de árvores em ambiente fabril real. [Computers & Industrial Engineering, 184, 109566](https://doi.org/10.1016/j.cie.2023.109566).
    
### Lacuna defendida pelo paper
Entre os trabalho encontrados, não foi identificado uma implementação que reúna simultaneamente:

- previsão diária da quantidade de recorrências rápidas de alarme;
- definição de alvo por regimes de cascata (IAT < 30min);
- registros reais de uma planta automotiva *Body-in-White*;
- comparação de LSTM e MLP nos mesmos conjuntos simples e completo;
- validação expansiva e teste temporal final;
- comparação direta com Random Forest, XGBoost e persistência (*baseline*)

A combinação desses componentes possibilitaram uma pesquisa que atinge uma lacuna específica e defensável, em que a novidade reside principalmente na formulação pioneira do alvo, na base real de alarmes, na comparação controlada dos conjuntos de atributos e na avaliação temporal comum a modelos tabulares e sequenciais.
