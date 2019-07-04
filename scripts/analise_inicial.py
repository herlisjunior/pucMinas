import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
#Carrega os dados
saeb_dados = pd.read_csv('pucMinas/dados/SAEB_ESCOLA.csv', sep = ',')
pd.set_option('display.max_rows', None)
saeb_dados.dtypes
saeb_dados.shape

#Seleciona as variáveis de interesse
saeb = saeb_dados.iloc[:,list(range(13)) + [79, 80]]
saeb = saeb.drop(columns = ['PC_FORMACAO_DOCENTE_FINAL', 'PC_FORMACAO_DOCENTE_MEDIO'])
saeb.dtypes

#Estruturação e análise básica dos dados
saeb.loc[:,'NIVEL_SOCIO_ECONOMICO'].head()
saeb.loc[:,'NIVEL_SOCIO_ECONOMICO'] = saeb.loc[:,'NIVEL_SOCIO_ECONOMICO'].astype('category')
saeb.isna().sum()

#Retirar as escolas que não possuem nota, pois nota é a variável resposta.
saeb = saeb[saeb.loc[:,'MEDIA_5EF_MT'].notnull()]
saeb.isna().sum()

#Devido a quantidade de registros, retirar dados NA de algumas variáveis.
saeb = saeb[saeb.loc[:,'PC_FORMACAO_DOCENTE_INICIAL'].notnull()]
saeb = saeb[saeb.loc[:,'NU_MATRICULADOS_CENSO_5EF'].notnull()]
saeb = saeb[saeb.loc[:,'TAXA_PARTICIPACAO_5EF'].notnull()]
saeb.isna().sum()

#Retirar escola em que todos faltaram
saeb = saeb[saeb.loc[:,'TAXA_PARTICIPACAO_5EF'] != 0]

#Análise Inicial
saeb.dtypes
saeb.describe().transpose()

#Gráficos para análise
saeb[['MEDIA_5EF_LP']].plot(kind = 'hist')
plt.show()
saeb[['MEDIA_5EF_MT']].plot(kind = 'hist')
plt.show()