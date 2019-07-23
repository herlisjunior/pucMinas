import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import statsmodels.api as sm
#Carregar Dados
censo_escolas = pd.read_csv('pucMinas/dados/ESCOLAS.csv', sep = '|', encoding='latin', dtype={"DT_ANO_LETIVO_INICIO": object, "DT_ANO_LETIVO_TERMINO": object})
pd.set_option('display.max_rows', None)
#Selecionar variáveis de interesse
censo_escolas = censo_escolas.iloc[:,[1, 4, 5, 6, 11, 13, 14, 15] + list(range(26, 105)) + list(range(120, 140))]
censo_escolas = censo_escolas[censo_escolas['TP_SITUACAO_FUNCIONAMENTO'] == 1]
censo_escolas = censo_escolas.drop(columns=['TP_OCUPACAO_GALPAO', 'TP_INDIGENA_LINGUA', 'CO_LINGUA_INDIGENA'])
#Analise Inicial
censo_escolas.shape
censo_escolas.dtypes
censo_escolas.describe().transpose()
censo_escolas.head().transpose()
censo_escolas['IN_LOCAL_FUNC_PREDIO_ESCOLAR'].value_counts()
censo_escolas['IN_LOCAL_FUNC_PRISIONAL_SOCIO'].value_counts()
pd.crosstab(censo_escolas['IN_LOCAL_FUNC_PREDIO_ESCOLAR'],censo_escolas['IN_LOCAL_FUNC_CASA_PROFESSOR'])
pd.crosstab(censo_escolas['TP_DEPENDENCIA'],censo_escolas['IN_LABORATORIO_CIENCIAS'], normalize='index')
pd.crosstab(censo_escolas['TP_DEPENDENCIA'],censo_escolas['IN_AUDITORIO'], normalize='index')


#Dados Faltantes
censo_escolas.isna().sum()[censo_escolas.isna().sum() != 0]

