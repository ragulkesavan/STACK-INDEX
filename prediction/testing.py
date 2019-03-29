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
y = df[['python']]
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(0, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
pred_uc = results.get_forecast(steps=60)
pred_ci = pred_uc.conf_int()
print(pred_uc.predicted_mean)
ax = y.plot(label='observed', figsize=(20, 15))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
            pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Year')
ax.set_ylabel('community support')
plt.legend()
plt.show()