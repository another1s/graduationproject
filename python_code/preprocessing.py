import csv
from readexcel import share_data
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import array, append, vstack, transpose, reshape,dot, true_divide, mean, exp, sqrt, log,loadtxt, savetxt, zeros, frombuffer
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs



path1 = "E:\onedrive\graduation\database\\returnoninvestedcapital"
path2 = "E:\onedrive\graduation\database\\grossprofitmargin"
share = share_data()


def loadfile(path):
    stats = dict()
    files = os.listdir(path)
    for filename in files:
        data0 = share.data(path, filename)
        stats[filename] = data0
 #   print(stats)
    return stats

def average_(data):
    sum_ = float(0)
    for i in data:
        #print(i)
        if (i[1] == 'Return on Invested Capital (TTM)')or (i[1]=='Gross Profit Margin (Quarterly)' or (i[1]=='') or i[1]=='None'):
            continue
        else :
            sum_ = sum_ +float(i[1])
   # print(sum_)
    average_sum = float(sum_ / len(data))
    #print(average_sum)
    return average_sum


ROE = loadfile(path1)
GPM = loadfile(path2)
for value in ROE.keys():
    ROE[value] = average_(ROE[value])
for value in GPM.keys():
    GPM[value] = average_(GPM[value])
print(ROE)
print(GPM)
length = len(ROE)
print(length)
i=0
data=list()
R=list()
G=list()
for i in ROE.values():
    R.append(i)
for j in GPM.values():
    G.append(j)
print(R)
print(G)
k = 0
while k <length:
    m=list()
    m.append(R[k])
    m.append(G[k])
    data.append(m)
    k = k + 1
print(data)
estimator = KMeans(n_clusters = 3)

feature_data = np.asarray(data)
#print(feature_data)
estimator.fit(feature_data)

path3 ="E:\onedrive\graduation\database\\testric"
path4 ="E:\onedrive\graduation\database\\testgpm"
testROE = loadfile(path3)
testGPM = loadfile(path4)
for value in testROE.keys():
    testROE[value] = average_(testROE[value])
for value in testGPM.keys():
    testGPM[value] = average_(testGPM[value])
print(testROE)
print(testGPM)
length = len(testROE)
print(length)
i=0
data0=list()
testR=list()
testG=list()
for i in testROE.values():
    testR.append(i)
for j in testGPM.values():
    testG.append(j)
print(testR)
print(testG)
k = 0
while k <length:
    m=list()
    m.append(testR[k])
    m.append(testG[k])
    data0.append(m)
    k = k + 1



print(estimator.fit_predict(feature_data))
test_data = np.asarray(data0)
print(estimator.predict(test_data))
print("hello")
label=estimator.fit_predict(feature_data)
label_pred = estimator.labels_ #获取聚类标签
centroids = estimator.cluster_centers_ #获取聚类中心
print(centroids)
inertia = estimator.inertia_ # 获取聚类准则的总和
mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
for i in range(50):
    plt.xlabel("return on in vested capital")
    plt.ylabel("gross profit margin")
    plt.plot(feature_data[i][0],feature_data[i][1],mark[label_pred[i]])
    print(label_pred[i])
#plt.xlim(-5, 5)
a=list()
b=list()
c=list()
d=list()
e=list()
for m in range(50):
    if label_pred[m] == 0:
        a.append(m)
    elif label_pred[m]==1:
        e.append(m)
    elif label_pred[m]==2:
        d.append(m)
    elif label_pred[m]==3:
        c.append(m)
    else :
        b.append(m)
#print("hello")
print(a)
print(b)
print(c)
print(d)
print(e)
share_sample=dict()
num=0
for name in ROE.keys():
    i=0
    while name[i] !='_':
        i=i+1
    print(name[0:i])
    share_sample[name[0:i]]=label[num]
    num=num+1
#print(share_sample)


def read_eps(path):
    eps_min = 0
    eps_max = 0
    eps = share_data()
    d = dict()
    path = path
    files = os.listdir(path)
    for filename in files:
        #  print(filename)
        eps_value = eps.data(path, filename)
        #print(eps_value)
        value = list()
        for s in eps_value:
            if s[0] != 'Period':
                if float(s[1]) < eps_min:
                    eps_min = float(s[1])
                if float(s[1]) > eps_max:
                    eps_max = float(s[1])
                value.append(float(s[1]))
        d[filename] = value
    print(eps_min,eps_max)
    return d

training = read_eps("E:\onedrive\graduation\database\epsvalue")
t = list()
for names in training.keys():
    t.append(training[names])
trainingdata = np.asarray(t)

#rint(training)
print(trainingdata)
estimator0 = KMeans(n_clusters=5)
print(estimator0.fit_predict(trainingdata))
labels = estimator0.labels_
#print(estimator0.cluster_centers_)
x = np.arange(3,63,3)
color = ['red', 'yellow', 'green', 'black', 'pink']
print(estimator0.cluster_centers_)

#for j in range(50):
    #w = array([trainingdata[j]])
    #print(w)
    #if labels[j]==2:
       # plt.xlabel("months")
       # plt.ylabel("PE")
       # plt.plot(x,w[0],color[labels[j]])
#0E 1C 2 A 3D 4B
plt.show()