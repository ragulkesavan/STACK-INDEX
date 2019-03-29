import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

df = pd.read_csv('multiTimeline.csv', skiprows=0)
df.columns = ['month','struts','spring','angular','reactjs','maven','gradle','postman','soapui','nltk','spacy','stanford-nlp','sql-server','mysql','mongodb','postgresql','java','python','html','css','.net','json','ruby','django','string','r','regex','ajax','bonecp','hikaricp','numpy','scipy','ruby-on-rails','hibernate','matlab','firefox','netbeans','php','linux','ios','android','eclipse','scala','visual-studio','macos','vb.net','windows','swing','amazon-web-services','selenium','spring-mvc','android-studio','hadoop','tomcat','jpa','javascript','c#','jquery','wordpress','ubuntu','swift','groovy','machine-learning','unix','jdbc','asp.net','wpf','flutter','bash','git','pandas','qt','cordova','codeigniter','symfony','xamarin','appcelerator','titanium','gcc','c','raspberry-pi','flash','curl','visual-c++','drupal','cron','angular5','xcode','shell','firebase','ms-access','cocoa','tcp','kotlin','xml','bitmap','docker','perl','pytorch','keras']
df.month = pd.to_datetime(df.month)
df.set_index('month', inplace=True)
y = df[['php']]
# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue