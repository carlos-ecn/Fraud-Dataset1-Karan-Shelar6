Projeto: Sistema Preditivo de Fraude Financeira

Elaborador: Carlos Eduardo Cruz Nakandakare

-----------------------------------------------------

Dataset Source:

Este projeto utiliza o dataset 'Fraud Dataset' disponível no Kaggle, criado por Karan-Shelar6.


=====================================

1. INTRODUÇÃO E OBJETIVO

=====================================

Identificação de Fraudes no Banco de Dados Financeiro, utilizando aprendizado de máquina. 

O foco é buscar melhor aproveitamento do Recall (detecção de fraude) e a Precisão (confiabilidade da previsão), 

e otimizar o trade-off entre Prejuízo (Falsos Negativos) e Experiência do Cliente (Falsos Positivos).


Desenvolvimento de um Sistema de Classificação de Fraude baseado em Time-Series Split, onde o modelo será treinado com 

transações passadas (step antigo) e validado em transações futuras (step futuro), simulando o ambiente real de produção com a 

chegada periódica de novos dados e clientes.


=====================================

2. FONTE E ESTRUTURA DOS DADOS

=====================================

Fraud Dataset
Karan Shelar6 · Updated 3 months ago

https://www.kaggle.com/datasets/karanshelar6/fraud-dataset


<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6362620 entries
Data columns (total 11 columns):
 #   Column          Dtype  
---  ------          -----  
 0   step            int64  
 1   type            object 
 2   amount          float64
 3   nameOrig        object 
 4   oldbalanceOrg   float64
 5   newbalanceOrig  float64
 6   nameDest        object 
 7   oldbalanceDest  float64
 8   newbalanceDest  float64
 9   isFraud         int64  
 10  isFlaggedFraud  int64  
dtypes: float64(5), int64(3), object(3)
memory usage: 534.0+ MB


=====================================

3. METODOLOGIA E ESTRATÉGIA DE MODELAGEM

=====================================

Pré-processamento: Limpeza, codificação (One-Hot Encoding).

Feature Engineering: Criação das colunas de agregação por cliente, transações sequenciais do histórico.

Aprendizado de Máquina e Modelagem: Escolha do XGBoost, Time-Series Split (Separação Cronológica) para validação.


=====================================

4. RESULTADOS FINAIS E AJUSTE CRÍTICO

=====================================

Raciocínio a partir dos resultados:

1. Escolha do XGBoost Vs Random Forest : 
- Precision (Confiabilidade) praticamente perfeito do RF levanta suspeitas;
- Recall (Detecção de Fraude) e F1 Score (Equilíbrio) são melhores e parecem compreender melhor o banco de dados;

2. Primeira performance XGBoost Randomized Search no Dataset futuro:
- Detecção de overfitting : Recall perfeito e Precision muito baixa, ou seja, Fraude detectadas em 100% e
muitos clientes que não são Fraude sendo classificados como Fraude;

3. Ajuste do Threshold:
- Maximizar o seu F1 Score, diminuir os casos de Falso Positivo, classificações erradas de Fraude;

4. Modelo final - XGBoost Randomized Search com Ajuste do Threshold:
Trade-Off controlado, aceitável e saudável:
- Queda de Falsos Positivos (FP), Para cada $100$ alertas de fraude, apenas $9$ são falsos.
- Classificação correta de Fraude: 84.52% dos casos, vazamento de 15% considerado controlável;

--------------------------

--- Performance no Histórico COMPLETO XGBoost Randomized Search

- Acurácia do modelo, verdadeiro positivo e verdadeiro negativo.
Accuracy: 0.9974 

- Precisão do modelo, falsos positivos. Confiabilidade.
Precision (Confiabilidade): 0.3286 

- Recall do modelo, classificação positiva de fraude verdadeira.
Recall (Detecção de Fraude): 1.0000

- F1-Score do modelo. Equilíbrio entre Precision e Recall.
F1 Score (Equilíbrio): 0.4947

- Avaliação de desempenho do modelo.
ROC AUC: 0.9999

--- Otimização do Threshold ---
Melhor Threshold (para F1 Score máximo): 0.9900

--- Performance RE-Ajustada (com Threshold Ótimo) ---
Novo F1 Score: 0.8776
Nova Precision: 0.9154
Novo Recall: 0.8428


=====================================

5. CONCLUSÃO

=====================================

O modelo final XGBoost, após otimização do Threshold para 0.9900, demonstrou um F1 Score de 0.8776,
resolvendo o problema de alta taxa de Falsos Positivos. A Precision de 91,54% garante alta confiabilidade nos alertas,
enquanto o Recall de 84,28% assegura a captura da maioria das fraudes. Este equilíbrio permite à empresa 
maximizar a economia de perdas (ROI), minimizando o impacto negativo na experiência do cliente.

=====================================

6. TECNOLOGIAS E FERRAMENTAS

=====================================

Python, Pandas, Scikit-learn, XGBoost, Joblib, Tableau.


=====================================

7. GUIA DE EXECUÇÃO E INSTALAÇÃO

=====================================


- Requisitos: Python 3.


- via pip:
pip install pandas numpy scikit-learn xgboost seaborn joblib kagglehub


- Execução: 
O projeto é apresentado em formato de notebook (ou script .py) e deve ser executado sequencialmente, 
iniciando pelo carregamento dos dados via API do Kaggle.


=====================================

8. RECURSOS E LINKS EXTERNOS

=====================================

Repositório GitHub: Acesse o código-fonte completo deste projeto.

https://github.com/carlos-ecn/Fraud-Dataset1-Karan-Shelar6.git

Dataset Kaggle:

https://www.kaggle.com/datasets/karanshelar6/fraud-dataset/data?select=Fraud.csv

Dashboard no Tableau Public: Visualize o dashboard interativo com os dados analisados.

https://public.tableau.com/views/ProjetoFraud/Painel1?:language=pt-BR&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link
