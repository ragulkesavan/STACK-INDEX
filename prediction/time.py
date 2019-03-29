import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf, pacf
sns.set()
df = pd.read_csv('multiTimeline.csv', skiprows=0)
df.columns = ['month','struts','spring','angular','reactjs','maven','gradle','postman','soapui','nltk','spacy','stanford-nlp','sql-server','mysql','mongodb','postgresql','java','python','html','css','.net','json','ruby','django','string','r','regex','ajax','bonecp','hikaricp','numpy','scipy','ruby-on-rails','hibernate','matlab','firefox','netbeans','php','linux','ios','android','eclipse','scala','visual-studio','macos','vb.net','windows','swing','amazon-web-services','selenium','spring-mvc','android-studio','hadoop','tomcat','jpa','javascript','c#','jquery','wordpress','ubuntu','swift','groovy','machine-learning','unix','jdbc','asp.net','wpf','flutter','bash','git','pandas','qt','cordova','codeigniter','symfony','xamarin','appcelerator','titanium','gcc','c','raspberry-pi','flash','curl','visual-c++','drupal','cron','angular5','xcode','shell','firebase','ms-access','cocoa','tcp','kotlin','xml','bitmap','docker','perl','pytorch','keras']
df.month = pd.to_datetime(df.month)
df.set_index('month', inplace=True)

overall=[]
for i in df.columns:
    diet=df[[i]]
    #overall.append(diet.rolling(12).mean())
    #diet.diff()
    overall.append(diet.diff())

df_rm = pd.concat(overall, axis=1)
df_rm.plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);


#diet.rolling(12).mean().plot(figsize=(20,10), linewidth=5, fontsize=20)
#plt.xlabel('Year', fontsize=20);

plt.show()