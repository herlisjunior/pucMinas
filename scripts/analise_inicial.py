import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

saeb_dados = pd.read_csv('pucMinas/dados/SAEB_ESCOLA.csv', sep = ',')
pd.set_option('display.max_rows', None)
saeb_dados.dtypes
saeb_dados.shape

saeb = saeb_dados.iloc[:,list(range(13)) + [79, 80]]
saeb.dtypes
saeb.loc[:,'NIVEL_SOCIO_ECONOMICO'].head()
saeb.loc[:,'NIVEL_SOCIO_ECONOMICO'] = saeb.loc[:,'NIVEL_SOCIO_ECONOMICO'].astype('category')
saeb.describe().transpose()
