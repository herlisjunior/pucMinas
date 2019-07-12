import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import statsmodels.api as sm
#Carrega os dados
saeb_dados = pd.read_csv('pucMinas/dados/SAEB_ESCOLA.csv', sep = ',')
pd.set_option('display.max_rows', None)
saeb_dados.dtypes
saeb_dados.shape

#Seleciona as variáveis de interesse
saeb = saeb_dados.iloc[:,list(range(13)) + [79, 80]]
saeb = saeb.drop(columns = ['PC_FORMACAO_DOCENTE_FINAL', 'PC_FORMACAO_DOCENTE_MEDIO'])
saeb.dtypes

#Estruturação dos dados
saeb.loc[:,'NIVEL_SOCIO_ECONOMICO'].head()
saeb.loc[:,'NIVEL_SOCIO_ECONOMICO'] = saeb.loc[:,'NIVEL_SOCIO_ECONOMICO'].astype('category')
saeb = saeb.replace({'ID_UF': {11:'RO', 12:'AC', 13:'AM', 14:'RR', 15:'PA', 16:'AP', 17:'TO', 21:'MA', 22:'PI', 23:'CE', 24:'RN', 25:'PB', 26:'PE', 27:'AL', 28:'SE', 29:'BA', 31:'MG', 32:'ES', 33:'RJ', 35:'SP', 41:'PR', 42:'SC', 43:'RS', 50:'MS', 51:'MT', 52:'GO', 53:'DF'}})
saeb.loc[:,'ID_UF'] = saeb.loc[:,'ID_UF'].astype('category')
saeb = saeb.replace({'ID_DEPENDENCIA_ADM': {1:'Federal', 2:'Estadual', 3:'Municipal', 4:'Privada'}})
saeb.loc[:,'ID_DEPENDENCIA_ADM'] = saeb.loc[:,'ID_DEPENDENCIA_ADM'].astype('category')
saeb = saeb.replace({'ID_LOCALIZACAO': {1:'Urbana', 2:'Rural'}})
saeb.loc[:,'ID_LOCALIZACAO'] = saeb.loc[:,'ID_LOCALIZACAO'].astype('category')
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
saeb.groupby('ID_UF')['MEDIA_5EF_LP', 'MEDIA_5EF_MT'].mean().sort_values(by = 'MEDIA_5EF_MT', ascending = False).plot( kind = 'bar')
plt.show()
saeb.groupby('ID_DEPENDENCIA_ADM')['MEDIA_5EF_LP', 'MEDIA_5EF_MT'].mean().sort_values(by = 'MEDIA_5EF_MT', ascending = False).plot( kind = 'bar')
plt.show()
saeb.groupby('ID_LOCALIZACAO')['MEDIA_5EF_LP', 'MEDIA_5EF_MT'].mean().sort_values(by = 'MEDIA_5EF_MT', ascending = False).plot( kind = 'bar')
plt.show()
saeb.plot(x = 'PC_FORMACAO_DOCENTE_INICIAL', y = 'MEDIA_5EF_LP', kind = 'scatter')
plt.show()
saeb.plot(x = 'PC_FORMACAO_DOCENTE_INICIAL', y = 'MEDIA_5EF_MT', kind = 'scatter')
plt.show()
saeb.plot(x = 'TAXA_PARTICIPACAO_5EF', y = 'MEDIA_5EF_LP', kind = 'scatter')
plt.show()
saeb.plot(x = 'TAXA_PARTICIPACAO_5EF', y = 'MEDIA_5EF_MT', kind = 'scatter')
plt.show()

#Criar dummys para UF, Dependência Adm e Localização
dummies = pd.get_dummies(saeb[['ID_UF', 'ID_DEPENDENCIA_ADM', 'ID_LOCALIZACAO']], drop_first= True)
dummies.shape
saeb.shape
saeb_dummies = pd.concat([saeb, dummies], axis= 1)


#Modelo OLS 01
saeb_LP = saeb_dummies[['MEDIA_5EF_LP']]
saeb_exog = saeb_dummies.drop(columns = ['ID_PROVA_BRASIL', 'ID_UF', 'ID_MUNICIPIO', 'ID_ESCOLA', 'ID_DEPENDENCIA_ADM', 'ID_LOCALIZACAO', 'NIVEL_SOCIO_ECONOMICO', 'NU_MATRICULADOS_CENSO_5EF', 'NU_PRESENTES_5EF', 'MEDIA_5EF_LP', 'MEDIA_5EF_MT', 'ID_DEPENDENCIA_ADM_Privada'])
saeb_exog = sm.add_constant(saeb_exog, prepend= False)
modelo01 = sm.OLS(saeb_LP, saeb_exog)
resultado01 = modelo01.fit()
resultado01.summary()

#Dados do PIB Municipal para estimar dados faltantes na categoria NIVEL_SOCIO_ECONOMICO
pib = pd.read_excel('pucMinas/dados/PIB dos Municípios - base de dados 2010-2016.xls', sheet_name = 'PIB_dos_Municípios')
pib.head().transpose()
pib = pib[pib['Ano'] == 2016]
pib = pib[['Nome da Unidade da Federação', 'Nome do Município', 'Produto Interno Bruto per capita\n(R$ 1,00)']]
pib.dtypes
pib['Chave'] = pib['Nome do Município'].str.cat(pib['Nome da Unidade da Federação'], sep = " ")

#Dados dos municipios do Saeb
municipios = pd.read_excel('pucMinas/dados/TS_MUNICIPIO.xlsx', sheet_name = 'TS_MUNICIPIO')
municipios.head().transpose()
municipios = municipios[['NO_MUNICIPIO', 'NO_UF', 'CO_MUNICIPIO']]
municipios['Chave'] = municipios['NO_MUNICIPIO'].str.cat(municipios['NO_UF'], sep = " ")


#Juntar os dados de pib e codigo do municipio
municipios = municipios.join(pib.set_index('Chave'), on = 'Chave', how = 'left')
municipios = municipios.sort_values('CO_MUNICIPIO')
municipios = municipios.drop_duplicates(subset = 'CO_MUNICIPIO', keep = 'first')
municipios.isna().sum()
municipios.head().transpose()
municipios.shape


#Juntar pib com dados saeb e tirar dados faltantes tanto para pib como para nivel socio economico
saeb_pib = saeb_dummies.join(municipios.set_index('CO_MUNICIPIO'), on = 'ID_MUNICIPIO', how = 'left')
saeb_pib.shape
saeb_pib = saeb_pib[(saeb_pib['Produto Interno Bruto per capita\n(R$ 1,00)'].notnull() | saeb_pib['NIVEL_SOCIO_ECONOMICO'].notnull())]
saeb_pib.isnull().sum()

#Separar dados para train e test para modelo de previsão de NIVEL_SOCIO_ECONOMICO
saeb_notnull = saeb_pib[(saeb_pib['Produto Interno Bruto per capita\n(R$ 1,00)'].notnull() & saeb_pib['NIVEL_SOCIO_ECONOMICO'].notnull())]
saeb_null = saeb_pib[saeb_pib['NIVEL_SOCIO_ECONOMICO'].isnull()]
train, test = train_test_split(saeb_notnull, test_size = 0.2)
trainX = train.drop(columns = ['ID_PROVA_BRASIL', 'ID_UF', 'ID_MUNICIPIO', 'ID_ESCOLA',
'ID_DEPENDENCIA_ADM', 'ID_LOCALIZACAO', 'PC_FORMACAO_DOCENTE_INICIAL',
'NIVEL_SOCIO_ECONOMICO', 'NU_MATRICULADOS_CENSO_5EF',
'NU_PRESENTES_5EF', 'TAXA_PARTICIPACAO_5EF', 'MEDIA_5EF_LP',
'MEDIA_5EF_MT', 'NO_MUNICIPIO', 'NO_UF', 'Chave',
'Nome da Unidade da Federação', 'Nome do Município',])

testX = test.drop(columns = ['ID_PROVA_BRASIL', 'ID_UF', 'ID_MUNICIPIO', 'ID_ESCOLA',
'ID_DEPENDENCIA_ADM', 'ID_LOCALIZACAO', 'PC_FORMACAO_DOCENTE_INICIAL',
'NIVEL_SOCIO_ECONOMICO', 'NU_MATRICULADOS_CENSO_5EF',
'NU_PRESENTES_5EF', 'TAXA_PARTICIPACAO_5EF', 'MEDIA_5EF_LP',
'MEDIA_5EF_MT', 'NO_MUNICIPIO', 'NO_UF', 'Chave',
'Nome da Unidade da Federação', 'Nome do Município',])

trainY = train[['NIVEL_SOCIO_ECONOMICO']]
testY = test[['NIVEL_SOCIO_ECONOMICO']]

#Modelo e estimativa de NIVEL_SOCIO_ECONOMICO com Decision Tree
arvore = tree.DecisionTreeClassifier()
arvore = arvore.fit(trainX, trainY)
arvore.score(testX, testY)
saeb_null['NIVEL_SOCIO_ECONOMICO'] = arvore.predict(saeb_null.drop(columns = ['ID_PROVA_BRASIL', 'ID_UF', 'ID_MUNICIPIO', 'ID_ESCOLA',
'ID_DEPENDENCIA_ADM', 'ID_LOCALIZACAO', 'PC_FORMACAO_DOCENTE_INICIAL',
'NIVEL_SOCIO_ECONOMICO', 'NU_MATRICULADOS_CENSO_5EF',
'NU_PRESENTES_5EF', 'TAXA_PARTICIPACAO_5EF', 'MEDIA_5EF_LP',
'MEDIA_5EF_MT', 'NO_MUNICIPIO', 'NO_UF', 'Chave',
'Nome da Unidade da Federação', 'Nome do Município',]))
saeb_null.head().transpose()
saeb_predito = pd.concat([saeb_null, saeb_pib[saeb_pib['NIVEL_SOCIO_ECONOMICO'].notnull()]])
saeb_predito.shape
saeb_predito.isnull().sum()
dummy_NSE = pd.get_dummies(saeb_predito['NIVEL_SOCIO_ECONOMICO'], drop_first=True)
saeb_predito = pd.concat([saeb_predito, dummy_NSE], axis = 1)


#Modelo OLS 02 - com NIVEL_SOCIO_ECONOMICO predito para escolas que não possuiam essa informação.
saeb_LP02 = saeb_predito[['MEDIA_5EF_LP']]
saeb_exog02 = saeb_predito.drop(columns = ['ID_PROVA_BRASIL', 'ID_UF', 'ID_MUNICIPIO', 'ID_ESCOLA',
'ID_DEPENDENCIA_ADM', 'ID_LOCALIZACAO', 'NIVEL_SOCIO_ECONOMICO', 'NU_MATRICULADOS_CENSO_5EF',
'NU_PRESENTES_5EF', 'MEDIA_5EF_LP', 'MEDIA_5EF_MT', 'ID_DEPENDENCIA_ADM_Privada', 'NO_MUNICIPIO',
'NO_UF', 'Chave', 'Nome da Unidade da Federação', 'Nome do Município',
'Produto Interno Bruto per capita\n(R$ 1,00)'])
saeb_exog02 = sm.add_constant(saeb_exog02, prepend= False)
modelo02 = sm.OLS(saeb_LP02, saeb_exog02)
resultado02 = modelo02.fit()
resultado02.summary()