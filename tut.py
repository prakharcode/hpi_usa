import quandl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn import svm, preprocessing, cross_validation
import numpy as np
#
# lis=[]
# with open('statename.txt','r') as handle:
# lis=handle.read().split(',')
#     print lis
# for i in lis[1:-1]:
#     print "FMAC/HPI_"+i
#     df = quandl.get("FMAC/HPI_"+i, authtoken = "LXMx3yx67VPbiTmwG5PJ")
#     df.to_csv(i+'data.csv')
# df = pd.read_csv('AKdata.csv', index_col='Date')
# for i in lis[2:-1]:
#     df2 = pd.read_csv(i+'data.csv',index_col='Date')
#     df[i+'_val'] = df2['Value']
# df.to_csv('finaldata.csv')
# for i in lis[2:-1]:
#     init = df[i+"_val"] [0]
#     df[i+"_val"] = ((df[i+"_val"] - init)/init)*100
# init = df["Value"] [0]
# df['Value'] = ((df['Value'] - init)/init)*100
# df.to_csv('finaldata.csv')
# df.index = pd.to_datetime(df.index)
# df2 = quandl.get("FMAC/MORTG",trim_start='1974-12-31', authtoken = "LXMx3yx67VPbiTmwG5PJ")
# df2 = df2.resample('D').sum()
# df2 = df2.resample('M').sum()
# print df2.head()
# df3 = df.join(df2)
# # df3['Value'].plot()
# # plt.show()
# df3.rename(columns = {'Value': 'M30'}, inplace = True)
# init = df3['M30'][0]
# df3['M30'] = ((df3['M30'] - init)/init)*100
# df3.to_csv('finaldata.csv')
# df2  = quandl.get("FMAC/HPI_USA", authtoken="LXMx3yx67VPbiTmwG5PJ")
# df['benchmark'] = df2['Value']
# df['US_Future'] = df['benchmark'].shift(-1)
# df ['label'] = list(map(lg, df['benchmark'],df['US_Future']))
# df.to_csv('finaldata.csv')
# def lg(cur, fut):
#     if cur<fut:
#         return 1
#     else:
#         return 0
df = pd.read_csv('finaldata.csv', index_col='Date')
X = np.array( df.drop( ['label' ,'US_Future'],1 ) )
X= preprocessing.scale(X)
y = np.array( df['label'] )
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf = svm.SVC(kernel = 'linear')
clf.fit(X_train,y_train)
print "The Accuracy is : %f" % (clf.score(X_test,y_test)*100)

df = df.drop(['M30','US_Future','benchmark','label'],1)
df.plot()
plt.legend().remove()
plt.show()
