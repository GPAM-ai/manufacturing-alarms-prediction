---
tags:
  - IC
  - UNIFESP
data: 2026-09-09
---
# Seções do Paper
---
## Orientações Gerais
Aqui constará uma descrição breve do que deve se esperar em cada região do paper e também orientações para o desenvolvimento do mesmo. 

Ele deve ser escrito em inglês. Estudar os maneirismos dos arquivos .md enviados para utilizar o estilo de escrita como base para a escrita do paper. Passar um filtro de formalismo e polimento, com vocabulário e construção de orações no nível esperado de uma pessoa graduanda em computação. Use o relatório feito em LaTeX como base também, a escrita dele está muito boa. Não insira no paper eventuais comentários que você direcionou a mim no processo de desenvolvimento. O paper deve documentar metodologia, referências, resultados e análises dos mesmos. 
## Seções
O número das seções deve ser marcado como (2. Related Works ou 3.1 Dataset). O sumário abaixo não é um definitivo, apenas um esboço que tenta abranger as ideias do trabalho. Altere os títulos conforme achar necessário para adequar ao contexto formal e científico do paper. Apenas avise caso hajam alterações.

0. Abstract
1. Introduction
	1. Context and Motivation
	2. Goals
		1. General Goals
		2. Specific Goals
2. Related Works
	1. Predictive Maintenance and Industrial Alarms
	2. Time Series and Predictive Modelling
	3. Comparable Experiments
	4. Lacuna de Pesquisa (traduzir)
3. Material and Methods
	1. Dataset Details
	2. Data Analysis Methodology
	3. Tools for Data Preparation
	4. Algorithm Fundamentals
		1. XGBoost & Random Forest
		2. LSTM
		3. MLP
		4. Grid Search & Optuna
	5. Experiment Validation
4. Methodology
	1. Preprocessing Data
		1. Temporal Aggregation
		2. Removing Outliers
	2. Exploratory Analysis
		1. Hiatus Diagnosis and Classification
		2. Recurrence Analysis for Machine and Shift
		3. Spectral Analysis by Fast Fourier Transform
		4. Additive Decomposition of Time Series
		5. Anomalies Detected
		6. Daily Series Autocorrelation
		7. Macroeconomic Exogenous Variables Correlation
	3. Feature Engineering
		1. Strategy and Context
		2. Time-related and Cyclical Features
		3. Rolling Window Features
		4. Lagged Daily Volume Features
		5. Recurrence Flags Features
		6. Hiatus Classification Features
		7. Machine related Features
		8. Exogenous Variables
	4. Predictive Modelling
		1. Target Definition and Time Horizon
		2. Validation Strategy
		3. Models, Experimental Configurations and Baseline
5. Experimental Results
	1. Expansive Validation Results
	2. Isolated Test Results
	3. Winner Model's most important Features
	4. Predictive Limitations
6. Final Remarks
	1. Discussions
		1. Scientific Contributions
		2. Practical Benefits for Industries
		3. Limitations and Threats to Validation
	2. Conclusions
	3. Future Works Propose
7. Acknowledgements
8. References

A seção de Acknowledgements deve agradecer à parceria com o proc. 2020/09850-0 da FAPESP, que forneceu os dados utilizados. Também cabe um agradecimento a CNPq pela bolsa de estudos que financiou esta pesquisa.

A seção de References deve conter todas as referências recicladas do relatório e também as novas definidas em [[paper_defined_refs]].

Future Works Propose deve propor uso de novas técnicas de validação, uso de redes neurais mais profundas que LSTM e investigação da utilidade preditiva dos dados descartados do dataset (Tratto, Billancela, Fault Codes).