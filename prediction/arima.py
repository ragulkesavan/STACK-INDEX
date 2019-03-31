import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
date=['2018-01-01','2018-02-01','2018-03-01','2018-04-01','2018-05-01','2018-06-01','2018-07-01','2018-08-01','2018-09-01','2018-10-01','2018-11-01','2018-12-01','2019-01-01','2019-02-01','2019-03-01','2019-04-01','2019-05-01','2019-06-01','2019-07-01','2019-08-01','2019-09-01','2019-10-01','2019-11-01','2019-12-01','2020-01-01','2020-02-01','2020-03-01','2020-04-01','2020-05-01','2020-06-01','2020-07-01','2020-08-01','2020-09-01','2020-10-01','2020-11-01','2020-12-01','2021-01-01','2021-02-01','2021-03-01','2021-04-01','2021-05-01','2021-06-01','2021-07-01','2021-08-01','2021-09-01','2021-10-01','2021-11-01','2021-12-01','2022-01-01','2022-02-01','2022-03-01','2022-04-01','2022-05-01','2022-06-01','2022-07-01','2022-08-01','2022-09-01','2022-10-01','2022-11-01','2022-12-01']
lis=['month','struts','spring','angular','reactjs','maven','gradle','postman','soapui','nltk','spacy','stanford-nlp','sql-server','mysql','mongodb','postgresql','java','python','html','css','.net','json','ruby','django','string','r','regex','ajax','bonecp','hikaricp','numpy','scipy','ruby-on-rails','hibernate','matlab','firefox','netbeans','php','linux','ios','android','eclipse','scala','visual-studio','macos','vb.net','windows','swing','amazon-web-services','selenium','spring-mvc','android-studio','hadoop','tomcat','jpa','javascript','c#','jquery','wordpress','ubuntu','swift','groovy','machine-learning','unix','jdbc','asp.net','wpf','flutter','bash','git','pandas','qt','cordova','codeigniter','symfony','xamarin','appcelerator','titanium','gcc','c','raspberry-pi','flash','curl','visual-c++','drupal','cron','angular5','xcode','shell','firebase','ms-access','cocoa','tcp','kotlin','xml','bitmap','docker','perl','pytorch','keras']
df1=pd.DataFrame(index=date,columns=lis)
df = pd.read_csv('multiTimeline.csv', skiprows=0)
df1=df1.fillna(0)
df.columns = ['month','struts','spring','angular','reactjs','maven','gradle','postman','soapui','nltk','spacy','stanford-nlp','sql-server','mysql','mongodb','postgresql','java','python','html','css','.net','json','ruby','django','string','r','regex','ajax','bonecp','hikaricp','numpy','scipy','ruby-on-rails','hibernate','matlab','firefox','netbeans','php','linux','ios','android','eclipse','scala','visual-studio','macos','vb.net','windows','swing','amazon-web-services','selenium','spring-mvc','android-studio','hadoop','tomcat','jpa','javascript','c#','jquery','wordpress','ubuntu','swift','groovy','machine-learning','unix','jdbc','asp.net','wpf','flutter','bash','git','pandas','qt','cordova','codeigniter','symfony','xamarin','appcelerator','titanium','gcc','c','raspberry-pi','flash','curl','visual-c++','drupal','cron','angular5','xcode','shell','firebase','ms-access','cocoa','tcp','kotlin','xml','bitmap','docker','perl','pytorch','keras']
df.month = pd.to_datetime(df.month)
df.set_index('month', inplace=True)
for i in df.columns:
    y = df[[i]]
    mod = sm.tsa.statespace.SARIMAX(y,
                                order=(0, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    results = mod.fit()
    pred_uc = results.get_forecast(steps=60)
    pred_ci = pred_uc.conf_int()
    print(i)
    print(pred_uc.predicted_mean)
    ax = y.plot(label='observed', figsize=(20, 15))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Year')
    ax.set_ylabel('community support')
    plt.legend()
    df1[i]=pred_uc.predicted_mean
    print(df1)

df1.to_csv('final.csv',sep='\t')
