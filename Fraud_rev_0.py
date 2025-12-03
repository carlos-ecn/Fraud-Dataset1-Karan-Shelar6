# %%

# ----------------------------------------------------------------------
# PROJETO DE DETECÇÃO DE FRAUDES EM TRANSAÇÕES FINANCEIRAS
# PROJECT FOR DETECTION OF FRAUDS IN FINANCIAL TRANSACTIONS
# ----------------------------------------------------------------------

# %%

# Importando as bibliotecas - Importing libraries

# Manipulação de Dados - Data Manipulation
import pandas as pd
import datetime as dt

# Cálculo Numérico e Estatístico - Numerical and Statistical Calculation
import numpy as np
import math as mth
from scipy import stats as st

# Visualização de Dados - Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

from plotly import graph_objects as go
import plotly.express as px

# Machine Learning e Pré-processamento - Machine Learning and Preprocessing

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# Modelos de Classificação
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import xgboost as xgb
from xgboost import XGBClassifier

# Modelos de Clusterização
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage # Para Clusterização Hierárquica
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering

# Métricas de Avaliação
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import make_scorer

# %%

# ----------------------------------------------------------------------
# FASE 1: CARREGAMENTO E EXPLORAÇÃO INICIAL DOS DADOS
# PHASE 1: LOADING AND INITIAL EXPLORATION OF DATA
# ----------------------------------------------------------------------

# %%

# Importando as bibliotecas do Kaggle - Importing libraries from Kaggle
# API com o Kaggle

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub.datasets import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "Fraud.csv"
# file_path = "https://www.kaggle.com/datasets/karanshelar6/fraud-dataset/data?select=Fraud.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "karanshelar6/fraud-dataset",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("Primeiras linhas do DataFrame:/n", df.head())

# %%

# Verificação de variáveis e tipos - Checking variables and types
df.info()

# %%

# Verificação estatística - Statistical check

df.describe()

# %%

# Otimização de tipos de dados - Data type optimization

# Mapeamento de tipos de dados - Data type mapping
dtype_map = {
    'step': np.int16,
    'type': 'category',
    'amount': np.float32,
    'nameOrig': str,
    'oldbalanceOrg': np.float32,
    'newbalanceOrig': np.float32,
    'nameDest': str,
    'oldbalanceDest': np.float32,
    'newbalanceDest': np.float32,
    'isFraud': np.int8,
    'isFlaggedFraud': np.int8
}

# Aplicação do mapeamento - Applying the mapping
df_principal = df.astype(dtype_map)
print(df_principal.info(memory_usage='deep'))

# %%

# Verificação de Nulos e Duplicados
# Null and Duplicate Check

print('nulos:\n', df.isna().sum())
print()
print('diplicados: ',df.duplicated().sum())

# %%

# ----------------------------------------------------------------------
# FASE 2: ANÁLISE EXPLORATÓRIA DE DADOS (EDA) E ENGENHARIA DE FEATURES
# PHASE 2: EXPLORATORY DATA ANALYSIS (EDA) AND FEATURE ENGINEERING
# ----------------------------------------------------------------------

# %%

# Análise da "linha temporal" - Analysis of the "time line"

df['step'].value_counts().sort_index(ascending=True)
# df['step'].nunique()

# %%

# Análise dos Tipos de Transações - Analysis of Transaction Types
df['type'].value_counts()

# %%

# Análise do desbalanceamento de classes - Class imbalance analysis

df['isFraud'].value_counts()
df['isFlaggedFraud'].value_counts()

# %%

# Gráfico da "linha temporal" - "Time line" graph

sns.countplot(data=df, x='step')
plt.show()

# %%

# Análise gráfica de Fraudes ao longo do tempo - Graphical analysis of Frauds over time

df_fraud = df[df['isFraud'] == 1]
sns.countplot(data=df_fraud, x='step', hue='isFraud')
plt.show()

# %%

# ----------------------------------------------------------------------
# 2.1: ENGENHARIA DE FEATURES
# 2.1: FEATURE ENGINEERING
# ----------------------------------------------------------------------

# %%

'''
clients_group_orig - DataFrame com análise agregada dos clientes de origem - DataFrame with aggregated analysis of origin clients
clients_group_dest - DataFrame com análise agregada dos clientes de destino - DataFrame with aggregated analysis of destination clients
df_seq_num_orig - DataFrame com sequência numérica de transações por cliente de origem - DataFrame with numerical sequence of transactions by origin client
'''

# %%

# Concatenação do Datafreame com ele mesmo para análise comparativa - Concatenation of the Dataframe with itself for comparative analysis

df_df_concat = pd.concat([df_principal, df_principal], axis=1)
df_df_concat.columns = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
       'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud',
       'isFlaggedFraud', 'step2', 'type2', 'amount2', 'nameOrig2', 'oldbalanceOrg2',
       'newbalanceOrig2', 'nameDest2', 'oldbalanceDest2', 'newbalanceDest2',
       'isFraud2', 'isFlaggedFraud2']
print(df_df_concat.columns)

# %%

# Análise dos clientes de origem e destino - Analysis of origin and destination clients

clients_group_orig = df_df_concat.groupby('nameOrig').agg({'amount':'mean','amount2':'max','step':'count','type':'nunique','oldbalanceOrg':'last','newbalanceOrig':'last','oldbalanceDest':'last','newbalanceDest':'last','isFraud':'sum','isFlaggedFraud':'sum'}).reset_index()
clients_group_orig.columns = ['nameOrig','amount_mean','amount_max','transaction_count','type_nunique','oldbalanceOrg_last','newbalanceOrig_last','oldbalanceDest_last','newbalanceDest_last','isFraud_sum','isFlaggedFraud_sum']
# clients_group_orig['name'] = 'nameOrig'
print(clients_group_orig)

# Análise dos clientes de origem e destino - Analysis of origin and destination clients

clients_group_dest = df_df_concat.groupby('nameDest').agg({'amount':'mean','amount2':'max','step':'count','type':'nunique','oldbalanceOrg':'last','newbalanceOrig':'last','oldbalanceDest':'last','newbalanceDest':'last','isFraud':'sum','isFlaggedFraud':'sum'}).reset_index()
clients_group_dest.columns = ['nameDest','amount_mean','amount_max','transaction_count','type_nunique','oldbalanceOrg_last','newbalanceOrig_last','oldbalanceDest_last','newbalanceDest_last','isFraud_sum','isFlaggedFraud_sum']
# clients_group_dest['name'] = 'nameDest'
print(clients_group_dest)

# %%

# Criação de variável de sequência numérica de transação por cliente - Creating a numerical sequence variable of transactions by client

# df de suporte
df_name_step = df_principal[['nameOrig','nameDest','step']]

# Agrupamento por cliente de origem - Grouping by origin client

df_seq_num_orig = df_name_step.sort_values(by=['nameOrig','step'])
df_seq_num_orig['seq_num'] = df_seq_num_orig.groupby('nameOrig').cumcount() + 1
print(df_seq_num_orig)

# Agrupamento por cliente de destino - Grouping by destination client

# df_seq_num_dest = df_name_step.sort_values(by=['nameDest','step'])
# df_seq_num_dest['seq_num'] = df_seq_num_dest.groupby('nameDest').cumcount() + 1
# print(df_seq_num_dest)

# %%

# ----------------------------------------------------------------------
# 2.2: UNIÃO DAS FEATURES AO DATAFRAME PRINCIPAL
# 2.2: UNION OF FEATURES TO THE MAIN DATAFRAME
# ----------------------------------------------------------------------

# %%

# União dos DataFrames de apoio com o DataFrame principal - Joining the support DataFrames with the main DataFrame
# Adição das novas Features ao DF principal - Adding new Features to the main DF

# %%

# Merge das novas Features com o DataFrame principal - Merging new Features with the main DataFrame

# Primeira etapa de merge com os dados agrupados dos clientes de origem - First merge step with aggregated data from origin clients
features_para_merge_orig = ['nameOrig','amount_mean','amount_max','transaction_count','type_nunique','isFraud_sum','isFlaggedFraud_sum']

df_features_apoio = pd.merge(df_principal, clients_group_orig[features_para_merge_orig], how='left', on='nameOrig', suffixes=('_principal', '_orig'))

# Segunda etapa de merge com os dados agrupados dos clientes de destino - Second merge step with aggregated data from destination clients
features_para_merge_dest = ['nameDest','amount_mean','amount_max','transaction_count','type_nunique','isFraud_sum','isFlaggedFraud_sum']

df_features = pd.merge(df_features_apoio, clients_group_dest[features_para_merge_dest], how='left', on='nameDest', suffixes=('_orig','_dest'))

# Visualização do DataFrame com as novas Features - Visualization of the DataFrame with the new Features
print(df_features.head())

df_features.columns

# %%

# Merge de Features de transações sequenciais por cliente de origem e destino - Merge of sequential transaction Features by origin and destination client

# Primeira etapa de merge com os dados de sequência numérica dos clientes de origem - First merge step with numerical sequence data from origin clients
df_features_apoio_2 = pd.merge(df_features, df_seq_num_orig[['nameOrig','step','seq_num']], how='left', on=['nameOrig','step'], suffixes=('_principal', '_orig'))

# Segunda etapa de merge com os dados de sequência numérica dos clientes de destino - Second merge step with numerical sequence data from destination clients
# df_features_2 = pd.merge(df_features_apoio_2, df_seq_num_dest[['nameDest','step','seq_num']], how='left', on=['nameDest','step'], suffixes=('_orig','_dest'))

# print(df_features_2.head())
# df_features_2.columns

print(df_features_apoio_2.head())
df_features_apoio_2.columns


# %%

# Otimização de tipos de dados - Data type optimization

# Mapeamento de tipos de dados - Data type mapping
dtype_map_2 = {
    'step': np.int16,
    'type': 'category',
    'amount': np.float32,
    'nameOrig': str,
    'oldbalanceOrg': np.float32,
    'newbalanceOrig': np.float32,
    'nameDest': str,
    'oldbalanceDest': np.float32,
    'newbalanceDest': np.float32,
    'isFraud': np.int8,
    'isFlaggedFraud': np.int8,
    'amount_mean_orig': np.float32, 
    'amount_max_orig': np.float32,
    'transaction_count_orig': np.int8, 
    'type_nunique_orig': np.int8, 
    'isFraud_sum_orig': np.int8,
    'isFlaggedFraud_sum_orig': np.int8, 
    'amount_mean_dest': np.float32, 
    'amount_max_dest': np.float32,
    'transaction_count_dest': np.int8, 
    'type_nunique_dest': np.int8, 
    'isFraud_sum_dest': np.int8,
    'isFlaggedFraud_sum_dest': np.int8,
    'seq_num': np.int8,
    # 'seq_num_orig': np.int8, 
    # 'seq_num_dest': np.int8
}

# Aplicação do mapeamento - Applying the mapping
df_principal = df_features_apoio_2.astype(dtype_map_2)
print(df_principal.info(memory_usage='deep'))

# %%

# ----------------------------------------------------------------------
# 2.3: VISUALIZAÇÃO GRAFICA GERAL
# 2.3: DATA VISUALIZATION
# ----------------------------------------------------------------------

# %%
# %%
# Análise de correlação entre as variáveis - Correlation analysis between variables

df_correlacao = df_principal.drop(columns=['type','nameOrig','nameDest'])
img_correlacao = df_correlacao.corr()

plt.figure(figsize=(8,6))
sns.heatmap(img_correlacao, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.show()


# %%

# Verificação das distribuições de valores nos tipos de transações - Checking the distributions of values in transaction types

plt.figure(figsize=(8,6))
sns.boxplot(data=df_principal, x='type', y='amount')
plt.show()

# %%

# Análise gráfica da "linha temporal" de valores por tipo de transação - Graphical analysis of the "time line" of values by transaction type

plt.figure(figsize=(8,6))
sns.lineplot(data=df_principal, x='step', y='amount', hue='type')
plt.show()

# %%

# Análise gráfica da "linha temporal" em quantidade por tipo de transação - Graphical analysis of the "time line" in quantity by transaction type

plt.figure(figsize=(8,6))
sns.countplot(data=df_principal, x='step', hue='type')
plt.show()


# %%

# ----------------------------------------------------------------------
# FASE 3: CIÊNCIA DE DADOS - APRENDIZADO DE MÁQUINA
# PHASE 3: DATA SCIENCE - MACHINE LEARNING
# ----------------------------------------------------------------------

# %%

# ----------------------------------------------------------------------
# 3.1: PREPARAÇÃO DOS DADOS PARA O APRENDIZADO DE MÁQUINA
# 3.1: DATA PREPARATION FOR MACHINE LEARNING
# ----------------------------------------------------------------------

# %%

df_final = df_principal.copy()

# %%

# Convertendo 'type' em colunas binárias, utilização do One-Hot Encoding para a variável categórica 'type' - 
# Converting 'type' into binary columns, using One-Hot Encoding for the categorical variable 'type'

df_final = pd.get_dummies(df_final, columns=['type'], prefix='type', drop_first=True)

# Recuperação da coluna 'type' original para o DataFrame Final - Recovery of the original 'type' column for the Final DataFrame
  
df_apoio_type = df_principal[['step','nameOrig','type']].copy()

df_check_type = df_final.drop(columns=['type'], errors='ignore')

df_final = pd.merge(df_check_type, df_apoio_type, how='left', on=['nameOrig', 'step'])


# %%

# DEFINIÇÃO DA LINHA DE CORTE DO TEMPO
# Usando o cálculo de 70% dos steps únicos (do seu código original)

num_unique_steps = len(df_final['step'].unique())
corte_step_limite = df_final['step'].unique()[int(num_unique_steps * 0.7)]

print(f"Número total de steps únicos: {num_unique_steps}")
print(f"Step de corte para 70% (Treino) / 30% (Teste): {corte_step_limite}")


# %%

'''
Divisão da base original, onde 30% dos dados mais recentes (futuros) 
serão utilizados para teste do modelo, e 70% dos dados mais antigos (passados) 
serão utilizados para treino do modelo. - 
Splitting the original database, where 30% of the most recent (future) 
data will be used for model testing, and 70% of the oldest (past) 
data will be used for model training.

df_split_future - DataFrame com os dados futuros para teste do modelo -
DataFrame with future data for model testing

df_ml - DataFrame com os dados passados para treino do modelo -
DataFrame with past data for model training

Aplicação dos métodos de ML de classificação XGBoost, Random Forest no DataFrame df_ml -
Applying XGBoost, Random Forest classification ML methods on the df_ml DataFrame

'''
# %%

# Divisão do dataset em futuro e passado - Splitting the dataset into future and past

# Treino (Dados Passados)
df_ml = df_final[df_final['step'] < corte_step_limite]
# Teste (Dados Futuros - Simulação Real)
df_split_future = df_final[df_final['step'] >= corte_step_limite]

print(f"Tamanho do conjunto de Treino (Passado): {len(df_ml)} linhas")
print(f"Tamanho do conjunto de Teste (Futuro): {len(df_split_future)} linhas")

# %%

# Definição das variáveis preditoras e alvo e dos parâmetros - Defining predictor and target variables and parameters

X = df_ml.drop(columns=['isFraud','isFlaggedFraud','nameOrig','nameDest'])
y = df_ml['isFraud']
target = 'IsFraud'

colunas_para_remover = [
    'isFraud', 'isFlaggedFraud', 'nameOrig', 'nameDest', 'step','type',
    'isFraud_sum_orig', 'isFraud_sum_dest', 
    'isFlaggedFraud_sum_orig', 'isFlaggedFraud_sum_dest',
]

RANDOM_STATE = 42
TEST_SIZE = 0.3

# X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.3, random_state=42, )

# %%

"""
Exibe métricas comuns para avaliação de modelos de classificação binária.

Args:
    X_train (array): Dados de treino.
    X_test (array): Dados de teste.
    y_train (array): Rótulos de treino.
    y_test (array): Rótulos de teste.
"""

# Definição dos conjuntos de treino e teste e dos parâmetros - Defining training and testing sets and parameters

# X_train: Todas as colunas do df_ml (passado), exceto as de remoção
X_train = df_ml.drop(columns=colunas_para_remover)
# y_train: A coluna alvo 'isFraud' do df_ml
y_train = df_ml['isFraud']

# X_test: Todas as colunas do df_split_future (futuro), exceto as de remoção
X_test = df_split_future.drop(columns=colunas_para_remover)
# y_test: A coluna alvo 'isFraud' do df_split_future
y_test = df_split_future['isFraud']

print("\nColunas de Features (X) no Treino:")
print(X_train.columns)
print(f"\nVerificação da variável target no Treino: {y_train.name}")

# %%

# ----------------------------------------------------------------------
# 3.2: MODELO DE CLASSIFICAÇÃO XGBOOST COM TRATAMENTO DE DESEQUILÍBRIO
# 3.2: CLASSIFICATION MODEL XGBOOST WITH IMBALANCE TREATMENT
# ----------------------------------------------------------------------

# %%

# ----------------------------------------------------------------------
# 3.2.1: CÁLCULO DO WEIGHT PARA TRATAR DESEQUILÍBRIO
# 3.2.1: CALCULATION OF WEIGHT TO TREAT IMBALANCE
# ----------------------------------------------------------------------

# %%

# O XGBoost usa o parâmetro 'scale_pos_weight' para ajustar o desequilíbrio.
# Cálculo: (Total de amostras da classe 0) / (Total de amostras da classe 1)

contagem_0 = y_train.value_counts()[0]
contagem_1 = y_train.value_counts()[1]
scale_pos_weight = contagem_0 / contagem_1

print(f"\nContagem de Não-Fraude (0) no Treino: {contagem_0}")
print(f"Contagem de Fraude (1) no Treino: {contagem_1}")
print(f"Scale_Pos_Weight (Correção de Desequilíbrio): {scale_pos_weight:.2f}")


# %%

# ----------------------------------------------------------------------
# 3.2.2: APLICAÇÃO DO MODELO DE CLASSIFICAÇÃO XGBOOST
# 3.2.2: APPLICATION OF THE XGBOOST CLASSIFICATION MODEL
# ----------------------------------------------------------------------


# %%

"""
Exibe métricas comuns para avaliação de modelos de classificação binária.

Args:
    xgb_model (sklearn model): Instância de modelo a ser treinada.
    xgb_prediction (array): Previsões do modelo (classes).
    xgb_y_proba (array): Probabilidades estimadas para a classe positiva.
"""
# %%

xgb_model = XGBClassifier(
    # use_label_encoder=False, 
    eval_metric='logloss', 
    random_state=42, 
    scale_pos_weight=scale_pos_weight, # Parâmetro 'scale_pos_weight' para ajustar o desequilíbrio
    n_estimators=100, 
    learning_rate=0.1,
    )

# a. Treinamento do modelo - Model training

xgb_model.fit(X_train, y_train)

# b. Previsões do modelo - Model predictions

xgb_prediction = xgb_model.predict(X_test)

xgb_y_proba = xgb_model.predict_proba(X_test)[:, 1]


# %%

# ----------------------------------------------------------------------
# 3.2.3: AVALIAÇÃO DO MODELO DE CLASSIFICAÇÃO XGBOOST
# 3.2.3: EVALUATION OF THE XGBOOST CLASSIFICATION MODEL
# ----------------------------------------------------------------------

# %%

"""
Exibe métricas comuns para avaliação de modelos de classificação binária.

Args:
    xgb_accuracy (array): Acurácia do modelo, verdadeiro positivo e verdadeiro negativo.
    xgb_precision (array): Precisão do modelo, falsos positivos. Confiabilidade.
    xgb_recall (array): Recall do modelo, classificação positiva de fraude verdadeira.
    xgb_f1 (array): F1-Score do modelo. Equilíbrio entre Precision e Recall.
    xgb_roc_auc (array): avaliação de desempenho do modelo.
"""

xgb_accuracy = accuracy_score(y_test, xgb_prediction)
xgb_precision = precision_score(y_test, xgb_prediction)
xgb_recall = recall_score(y_test, xgb_prediction)
xgb_f1 = f1_score(y_test, xgb_prediction)
xgb_roc_auc = roc_auc_score(y_test, xgb_y_proba)

print('\n--- Resultados no Conjunto de Teste (Futuro) ---')
print(f'XGBoost Classifier Accuracy: {xgb_accuracy:.2f}')
print(f'XGBoost Classifier Precision: {xgb_precision:.2f}')
print(f'XGBoost Classifier Recall: {xgb_recall:.2f}')
print(f'XGBoost Classifier F1 Score: {xgb_f1:.2f}')
print(f'XGBoost Classifier ROC AUC: {xgb_roc_auc:.2f}')


# %%

# c. Matriz de Confusão do modelo XGBoost Classifier - Confusion Matrix of the XGBoost Classifier model

fig, ax = plt.subplots(figsize=(6, 6))
cm = confusion_matrix(y_test, xgb_prediction, labels=xgb_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=xgb_model.classes_)
disp.plot(cmap='Blues', ax=ax)
ax.set_title('Matriz de Confusão (Teste - Futuro)')
plt.show()

# %%

# ----------------------------------------------------------------------
# 3.3: OTIMIZAÇÃO DE HIPERPARÂMETROS (GRIDSEARCHCV E RANDOMIZEDSEARCHCV) DO MODELO DE CLASSIFICAÇÃO XGBOOST COM TRATAMENTO DE DESEQUILÍBRIO
# 3.3: HYPERPARAMETER OPTIMIZATION (GRIDSEARCHCV AND RANDOMIZEDSEARCHCV) OF THE XGBOOST CLASSIFICATION MODEL WITH IMBALANCE TREATMENT
# ----------------------------------------------------------------------

# %%

# 3.3.1. Definindo a métrica de otimização - Defining the optimization metric
# O F1-Score é a métrica mais adequada para equilibrar Precision e Recall em bases desbalanceadas.

f1_scorer = make_scorer(f1_score)

# 3.3.2. Definindo o grid de parâmetros a serem testados - Defining the parameter grid to be tested
'''
TENTATIVA 1 COM GRIDSEARCHCV - FIRST ATTEMPT WITH GRIDSEARCHCV
param_grid - grid de parâmetros para GridSearchCV

TENTATIVA 2 COM RANDOMIZEDSEARCHCV - SECOND ATTEMPT WITH RANDOMIZEDSEARCHCV
param_distributions - distribuições de parâmetros para RandomizedSearchCV
'''

# param_grid = {
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.05, 0.1, 0.2],
#     'n_estimators': [50, 100, 200],
#     'gamma': [0, 0.5, 1] # Redução de perda mínima necessária para fazer uma partição de folha adicional.
# }

param_distributions = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200, 300],
    'gamma': [0, 0.1, 0.5, 1],
    'subsample': [0.7, 0.8, 0.9] # Proporção de amostras usadas para treinar cada árvore.
}

# %%

# ----------------------------------------------------------------------
# 3.3.3: APLICAÇÃO DOS MODELOS DE OTIMIZAÇÃO DE HIPERPARÂMETROS
# 3.3.3: APPLICATION OF HYPERPARAMETER OPTIMIZATION MODELS
# ----------------------------------------------------------------------

# %%

# # d.1. Configurando o modelo inicial com GridSearchCV (incluindo o scale_pos_weight) - Setting up the initial model with GridSearchCV (including scale_pos_weight)
# xgb_grid = XGBClassifier(
#     use_label_encoder=False,
#     eval_metric='logloss',
#     random_state=42,
#     scale_pos_weight=scale_pos_weight # Mantendo a correção de desequilíbrio
# )

# d.2. Configurando o modelo inicial com RandomizedSearchCV (incluindo o scale_pos_weight) - Setting up the initial model with RandomizedSearchCV (including scale_pos_weight)
xgb_rand = XGBClassifier(
    # use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=scale_pos_weight # Mantendo a correção de desequilíbrio
)

# e. Configurando o TimeSeriesSplit - Setting up the TimeSeriesSplit
# Definimos 3 splits cronológicos, um padrão otimizado para validação temporal (n_splits=5 foi grande demais)
# - Defining 3 chronological splits, a standard optimized for temporal validation (n_splits=5 was too large)

tscv = TimeSeriesSplit(n_splits=3)

# %%

# # f.1. Configurando o GridSearchCV
# # cv=3 ou 5 para Cross-Validation (validação cruzada)

# grid_search = GridSearchCV(
#     estimator=xgb_grid,
#     param_grid=param_grid,
#     scoring=f1_scorer, 
#     cv=3, 
#     verbose=2, 
#     n_jobs=-1
# )

# f.2. Configurando o RandomizedSearchCV
# n_iter=30: Testará 30 combinações aleatórias.
# n_jobs=1: Crucial para evitar o MemoryError, limitando a um único processo.

random_search = RandomizedSearchCV(
    estimator=xgb_rand,
    param_distributions=param_distributions,
    scoring=f1_scorer, 
    cv=tscv,  # Aplicando o TimeSeriesSplit
    verbose=2, 
    n_jobs=1, # USAR n_jobs=1 OU 2 PARA EVITAR MEMORYERROR
    n_iter=15,
    random_state=42
)

# %%

# # g.1. Executando a Busca em Grid (Este processo pode levar alguns minutos)
# print("\nIniciando GridSearchCV para otimização de hiperparâmetros...")
# # Usamos X_train e y_train (os 70% passados)
# grid_search.fit(X_train, y_train)

# # g.2. Exibindo os melhores resultados
# print("\n--- Resultados do GridSearch ---")
# print(f"Melhores Hiperparâmetros: {grid_search.best_params_}")
# print(f"Melhor F1 Score (Validação Cruzada): {grid_search.best_score_:.4f}")

# %%

# h.1. Executando a Busca Aleatória (usando a base de treino) RandomizedSearchCV - (This process may take a few minutes)
print("\nIniciando RandomizedSearchCV para otimização de hiperparâmetros...")
# Usamos X_train e y_train (os 70% passados)
random_search.fit(X_train, y_train)

# h.2 Exibindo os melhores resultados RandomizedSearchCV - Displaying the best results RandomizedSearchCV
print("\n--- Resultados do RandomizedSearch ---")
print(f"Melhores Hiperparâmetros: {random_search.best_params_}")
print(f"Melhor F1 Score (Validação Cruzada): {random_search.best_score_:.4f}")


# %%

# i.1. Definindo o melhor modelo encontrado
# xgb_model_otimizado = grid_search.best_estimator_

# %%

# i.2. Definindo o melhor modelo encontrado RandomizedSearchCV - Defining the best model found RandomizedSearchCV
xgb_model_otimizado = random_search.best_estimator_

# %%

# ----------------------------------------------------------------------
# 3.3.4: AVALIAÇÃO DO MODELO OTIMIZADO NO CONJUNTO DE TESTE (FUTURO)
# 3.3.4: EVALUATION OF THE OPTIMIZED MODEL ON THE TEST SET (FUTURE)
# ----------------------------------------------------------------------

"""
Exibe métricas comuns para avaliação de modelos de classificação binária.

Args:
    xgb_accuracy (array): Acurácia do modelo, verdadeiro positivo e verdadeiro negativo.
    xgb_precision (array): Precisão do modelo, falsos positivos. Confiabilidade.
    xgb_recall (array): Recall do modelo, classificação positiva de fraude verdadeira.
    xgb_f1 (array): F1-Score do modelo. Equilíbrio entre Precision e Recall.
    xgb_roc_auc (array): avaliação de desempenho do modelo.
"""

# j.1. Previsões do modelo OTIMIZADO no conjunto de teste - Optimized model predictions on the test set

xgb_prediction_otimizado = xgb_model_otimizado.predict(X_test)
xgb_y_proba_otimizado = xgb_model_otimizado.predict_proba(X_test)[:, 1]

# %%

# j.2. Avaliação final RandomizedSearchCV no conjunto de teste (futuro) - Final Evaluation RandomizedSearchCV on the test set (future)

xgb_accuracy_otimizado = accuracy_score(y_test, xgb_prediction_otimizado)
xgb_precision_otimizado = precision_score(y_test, xgb_prediction_otimizado)
xgb_recall_otimizado = recall_score(y_test, xgb_prediction_otimizado)
xgb_f1_otimizado = f1_score(y_test, xgb_prediction_otimizado)
xgb_roc_auc_otimizado = roc_auc_score(y_test, xgb_y_proba_otimizado)

print('\n--- Resultados FINAIS do Modelo RandomizedSearchCV (Teste - Futuro) ---')
print(f'Accuracy: {xgb_accuracy_otimizado:.4f}')
print(f'Precision (Confiabilidade): {xgb_precision_otimizado:.4f}')
print(f'Recall (Detecção de Fraude): {xgb_recall_otimizado:.4f}')
print(f'F1 Score (Equilíbrio): {xgb_f1_otimizado:.4f}')
print(f'ROC AUC: {xgb_roc_auc_otimizado:.4f}')

# %%

# j.3. Matriz de Confusão Otimizada RandomizedSearchCV - Optimized Confusion Matrix RandomizedSearchCV

fig, ax = plt.subplots(figsize=(6, 6))
cm_otimizado = confusion_matrix(y_test, xgb_prediction_otimizado, labels=xgb_model_otimizado.classes_)
disp_otimizado = ConfusionMatrixDisplay(confusion_matrix=cm_otimizado, display_labels=xgb_model_otimizado.classes_)
disp_otimizado.plot(cmap='Blues', ax=ax)
ax.set_title('Matriz de Confusão RandomizedSearchCV')
plt.show()

# %%

# ----------------------------------------------------------------------
# 3.4: MODELO DE CLASSIFICAÇÃO RAMDOM FOREST
# 3.4: RANDOM FOREST CLASSIFICATION MODEL
# ----------------------------------------------------------------------

# %%

"""

Args:
    rforest_model - Instância do modelo Random Forest Classifier.
    rforest_prediction - Previsões do modelo Random Forest.
    rforest_y_proba - Probabilidades estimadas para a classe positiva pelo modelo Random Forest.
"""

# 3.4.1. Definição do modelo Random Forest Classifier - Defining the Random Forest Classifier model

rforest_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 3.4.2. Treinamento do modelo Random Forest Classifier - Model training Random Forest Classifier

rforest_model.fit(X_train, y_train)

# 3.4.3. Previsões do modelo Random Forest Classifier - Model predictions Random Forest Classifier

rforest_prediction = rforest_model.predict(X_test)

rforest_y_proba = rforest_model.predict_proba(X_test)[:, 1]

# %%

# 3.4.4. Avaliação do modelo Random Forest Classifier - Model evaluation Random Forest Classifier
"""
Exibe métricas comuns para avaliação de modelos de classificação binária.

Args:
    rforest_accuracy (array): Acurácia do modelo, verdadeiro positivo e verdadeiro negativo.
    rforest_precision (array): Precisão do modelo, falsos positivos. Confiabilidade.
    rforest_recall (array): Recall do modelo, classificação positiva de fraude verdadeira.
    rforest_f1 (array): F1-Score do modelo. Equilíbrio entre Precision e Recall.
    rforest_roc_auc (array): avaliação de desempenho do modelo.
"""

rforest_accuracy = accuracy_score(y_test, rforest_prediction)
rforest_precision = precision_score(y_test, rforest_prediction)
rforest_recall = recall_score(y_test, rforest_prediction)
rforest_f1 = f1_score(y_test, rforest_prediction)
rforest_roc_auc = roc_auc_score(y_test, rforest_y_proba)

print('\n--- Resultados FINAIS do Modelo Otimizado (Teste - Futuro) ---')
print(f'Random Forest Classifier Accuracy: {rforest_accuracy:.4f}')
print(f'Random Forest Classifier Precision (Confiabilidade): {rforest_precision:.4f}')
print(f'Random Forest Classifier Recall (Detecção de Fraude): {rforest_recall:.4f}')
print(f'Random Forest Classifier F1 Score (Equilíbrio): {rforest_f1:.4f}')
print(f'Random Forest Classifier ROC AUC: {rforest_roc_auc:.4f}')


# %%

# 3.4.5. Matriz de Confusão Random Forest - Random Forest Confusion Matrix

fig, ax = plt.subplots(figsize=(6, 6))
cm_otimizado = confusion_matrix(y_test, rforest_prediction, labels=rforest_model.classes_)
disp_otimizado = ConfusionMatrixDisplay(confusion_matrix=cm_otimizado, display_labels=rforest_model.classes_)
disp_otimizado.plot(cmap='Blues', ax=ax)
ax.set_title('Matriz de Confusão Random Forest (Modelo Otimizado - Teste Futuro)')
plt.show()


# %%

# ----------------------------------------------------------------------
# FASE 4: APLICAÇÃO NO DATAFRAME FINAL (XGBOOST RANDOMIZED SEARCH)
# PHASE 4: APPLICATION ON THE FINAL DATAFRAME (XGBOOST RANDOMIZED SEARCH)
# ----------------------------------------------------------------------

# %%

# 4.1. PREPARAÇÃO DO DATASET FINAL - PREPARATION OF THE FINAL DATASET

# Definição de X e y (Usando TODOS os dados para o treinamento final) - Defining X and y (Using ALL data for final training)
# REMOVENDO as features que causam o vazamento de dados - REMOVING features that cause Data Leakage

colunas_para_remover = [
    'isFraud', 'isFlaggedFraud', 'nameOrig', 'nameDest', 'step',
    'isFraud_sum_orig', 'isFraud_sum_dest',
    'isFlaggedFraud_sum_orig', 'isFlaggedFraud_sum_dest'
]

X_all = df_final.drop(columns=colunas_para_remover, errors='ignore')
y_all = df_final['isFraud']

# 4.2. Cálculo do scale_pos_weight - Calculation of scale_pos_weight

contagem_0 = y_all.value_counts()[0]
contagem_1 = y_all.value_counts()[1]
scale_pos_weight = contagem_0 / contagem_1

# 4.3. Definição dos Melhores Hiperparâmetros (Randomized Search) - Defining the Best Hyperparameters (Randomized Search)

best_params = {
    'subsample': 0.9,
    'n_estimators': 200,
    'max_depth': 9,
    'learning_rate': 0.05,
    'gamma': 1
}

# %%

# 4.4. TREINAMENTO DO MODELO FINAL - FINAL MODEL TRAINING

print("Iniciando o treinamento do modelo final XGBoost Randomized Search (usando 100% dos dados)")

xgb_model_final = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1, # Usar todos os núcleos do processador
    **best_params 
)

# O treinamento é feito com TODOS os dados disponíveis para máxima performance em produção
xgb_model_final.fit(X_all, y_all)

print("Treinamento XGBoost Randomized Search final concluído .")

# %%

# 4.5. AVALIAÇÃO EM TODO O DATASET E FEATURE IMPORTANCE - EVALUATION ON THE ENTIRE DATASET AND FEATURE IMPORTANCE

# Avaliação no dataset COMPLETO (para verificar a performance geral no histórico) - Evaluation on the COMPLETE dataset (to check overall performance in history)
y_pred_all = xgb_model_final.predict(X_all)
y_proba_all = xgb_model_final.predict_proba(X_all)[:, 1]

final_accuracy = accuracy_score(y_all, y_pred_all)
final_precision = precision_score(y_all, y_pred_all)
final_recall = recall_score(y_all, y_pred_all)
final_f1 = f1_score(y_all, y_pred_all)
final_roc_auc = roc_auc_score(y_all, y_proba_all)

print('\n--- Performance no Histórico COMPLETO XGBoost Randomized Search')
print(f'Accuracy: {final_accuracy:.4f}')
print(f'Precision (Confiabilidade): {final_precision:.4f}')
print(f'Recall (Detecção de Fraude): {final_recall:.4f}')
print(f'F1 Score (Equilíbrio): {final_f1:.4f}')
print(f'ROC AUC: {final_roc_auc:.4f}')

# %%

# 4.6. Feature Importance do Modelo Final - Feature Importance of the Final Model

feature_importance = pd.Series(xgb_model_final.feature_importances_, index=X_all.columns)
feature_importance_sorted = feature_importance.sort_values(ascending=False).head(10)

print('\nTop 10 Importância de Features (Modelo Final):')
print(feature_importance_sorted)

# %%

# 4.7. Plot Feature Importance XGBoost

fig, ax = plt.subplots(figsize=(10, 6))
feature_importance_sorted.plot(kind='barh', ax=ax, color='teal')
ax.set_title('Top 10 Importância de Features (Modelo Final - Todos os Dados)')
ax.set_xlabel('Importância (Fator de Ganho do XGBoost)')
ax.set_ylabel('Feature')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_final.png')
plt.show()

# %%
'''
Comportamento estatisticamente anormal observado:
Encontrado um desbalanceamento significativo entre Precision e Recall no modelo final.
Para mitigar esse problema, será realizada uma otimização do threshold de decisão.
'''

# %%

# ----------------------------------------------------------------------
# 4.8. OTIMIZAÇÃO DO THRESHOLD DE DECISÃO PARA MAXIMIZAR F1 SCORE
# 4.8. OPTIMIZATION OF THE DECISION THRESHOLD TO MAXIMIZE F1 SCORE
# ----------------------------------------------------------------------

# %%

#  4.8.1. Encontrando o Threshold Ótimo para o F1 Score - Finding the Optimal Threshold for F1 Score

def find_best_threshold(y_true, y_probas):
    """Encontra o threshold que maximiza o F1 Score."""
    thresholds = np.linspace(0.01, 0.99, 100) # Testa 100 pontos entre 1% e 99%
    best_f1 = 0
    best_thresh = 0.5
    
    for t in thresholds:
        y_pred = (y_probas >= t).astype(int)
        current_f1 = f1_score(y_true, y_pred)
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_thresh = t
            
    return best_thresh, best_f1

# Calcular o threshold ideal usando as probabilidades no histórico completo - 
# Calculating the ideal threshold using probabilities on the complete history

best_threshold, max_f1_score = find_best_threshold(y_all, y_proba_all)

print(f'\n--- Otimização do Threshold ---')
print(f'Melhor Threshold (para F1 Score máximo): {best_threshold:.4f}')
print(f'F1 Score Máximo Encontrado: {max_f1_score:.4f}')


# %%

# 4.8.2. Re-avaliação do Modelo com o Novo Threshold - Re-evaluation of the Model with the New Threshold

y_pred_tuned = (y_proba_all >= best_threshold).astype(int)

# 4.8.3. Cálculo das Novas Métricas - Calculation of New Metrics

tuned_precision = precision_score(y_all, y_pred_tuned)
tuned_recall = recall_score(y_all, y_pred_tuned)

print('\n--- Performance RE-Ajustada (com Threshold Ótimo) ---')
print(f'Novo F1 Score: {max_f1_score:.4f}')
print(f'Nova Precision: {tuned_precision:.4f}')
print(f'Novo Recall: {tuned_recall:.4f}')


# %%

# 4.8.4. Visualização da Matriz de Confusão Re-Ajustada - Visualization of the Re-Tuned Confusion Matrix

cm = confusion_matrix(y_all, y_pred_tuned)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Não Fraude (0)', 'Fraude (1)'])

# Plot Matriz de Confusão
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(cmap=plt.cm.Blues, ax=ax)
ax.set_title(f'Matriz de Confusão (Threshold: {best_threshold:.4f})')
plt.show()

print("\n")

# %%

# ----------------------------------------------------------------------
# FASE 5: EXPORTAÇÃO DO MODELO TREINADO PRONTO PARA PRODUÇÃO
# PHASE 5: EXPORTING THE TRAINED MODEL READY FOR PRODUCTION
# ----------------------------------------------------------------------

# %%

# 5.1. Feature para comparação de versões do modelo - Feature for model version comparison

# Assumimos que 'df_final' (com todas as features), 'X_all', 'y_all' e 'xgb_model_final' 
# estão definidos e treinados a partir das etapas anteriores.
# Vamos fixar o melhor threshold encontrado:

best_threshold = 0.9900

# 5.2. Previsão de Probabilidades (Modelo já treinado) - Probability Prediction (Model already trained)
# Usando X_all (features do histórico completo)

y_proba_all = xgb_model_final.predict_proba(X_all)[:, 1]
# print(y_proba_all)

# 5.3. Criação da Coluna 'new_fraud' (Modelo) - Creating the 'new_fraud' Column (Model)
# Aplica o Threshold de 0.9900 para a decisão binária

df_final['new_fraud'] = (y_proba_all >= best_threshold).astype(int)

# %%

# 5.4. CRIAÇÃO DA COLUNA DE COMPARAÇÃO - CREATING THE COMPARISON COLUMN

# A coluna comparison_fraud ajudará a classificar o acerto do modelo - 
# The comparison_fraud column will help classify the model's accuracy

def classify_comparison(row):
    """Classifica o resultado da previsão do modelo em relação ao rótulo real."""
    if row['isFraud'] == 1 and row['new_fraud'] == 1:
        return 'True Positive (Acerto - Fraude)' # Acertou a fraude (Recall)
    elif row['isFraud'] == 0 and row['new_fraud'] == 0:
        return 'True Negative (Acerto - Não Fraude)' # Acertou o não-fraude
    elif row['isFraud'] == 0 and row['new_fraud'] == 1:
        return 'False Positive (Erro - Falso Alerta)' # Erro: Não-fraude marcado como Fraude (Precision)
    elif row['isFraud'] == 1 and row['new_fraud'] == 0:
        return 'False Negative (Erro - Fraude Perdida)' # Erro: Fraude real marcada como Não-fraude
    else:
        return 'Outro'

# Aplica a função de comparação - Applies the comparison function

df_final['comparison_fraud'] = df_final.apply(classify_comparison, axis=1)

# %%

# -----------------------------------------------------------
# 5.5. Exporatação do modelo para a próxima seção de aprendizado de máquina
# 5.5. Exporting the model for the next machine learning section
# -----------------------------------------------------------

import joblib

model_filename = 'xgb_model_final.joblib'
joblib.dump(xgb_model_final, model_filename)

print(f'\nModelo final XGBoost salvo em: {model_filename}')

# %%

# -----------------------------------------------------------
# 5.6. EXPORTAÇÃO PARA CSV (Utilização para Dasboard - TABLEAU)
# 5.6. EXPORT TO CSV (Use for Dashboard - TABLEAU)
# -----------------------------------------------------------

# Seleciona as colunas essenciais, originais e as novas colunas
colunas_export = [
    'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 
    'nameDest', 'oldbalanceDest', 'newbalanceDest', 
    'isFraud',                   # Rótulo real da base
    'new_fraud',                 # Rótulo predito pelo seu modelo
    'comparison_fraud',          # Resultado da comparação
    'transaction_count_orig',    # Feature importante de histórico
    'transaction_count_dest',
    'amount_mean_orig',
    'amount_mean_dest',

]

df_export = df_final[colunas_export].copy()

# 3.1. Exportar para CSV
csv_filename = 'df_ML_fraude_para_dashboard.csv'
df_export.to_csv(csv_filename, index=False, encoding='utf-8')

print(f'\nDataFrame de exportação salvo em: {csv_filename}')

# %%

# ----------------------------------------------------------------------
# 
# ----------------------------------------------------------------------

# %%