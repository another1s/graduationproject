import csv
from readexcel import *
import os
import numpy as np
from numpy import array, append, vstack, transpose, reshape,dot, true_divide, mean, exp, sqrt, log,loadtxt, savetxt, zeros, frombuffer
from numpy.linalg import norm, lstsq
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import optimize

# 从2013 年到 2017 年 5 个整年，四个季度数据 一共20 个时间点的数据
# 所以输入为维数为20维，20个元组，每个元组由季度的市盈率与市净率构成
# 利用确定学习，使得RBF神经网络能够跟随输入，所以对于每一类，将收敛到一个最优权值
# 藉由五种不同的权值跟随得出的在20个时间点内的图形，与新输入的样本做差，最接近的则是分类结果

'''
    initiate constant variable 
    the number of neurons is determined by 20( 20 seasoning pair of data)
    each of them has (eps_value, pb_value)
    they range from eps_value_min to eps_value_max and pb_value_min to pb_value_max
    let the interval be the distance between each point
    so the num_neurons should be 20 * (max-min)/interval *(max-min)/interval
    
    weights matrix's dimension is as same as num_neurons
'''
num_training_data = 50
num_input_dimension = 20
num_output_dimension = 20
interval = 0.1
num_neurons = 7000
weights_matrix = 7000   # 20 * 350
eps_value_scale = 35  # -8.55 min 24.64 max
pb_value_scale = 40


'''
    RBF network,it should be three layers
    an input layer (20 points) and a output layer  
    a output layer which has same dimension as input layer
'''


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
class d_learing:
    def __init__(self):
        self.gamma = 0.1
        self.sigma = 0.02
        self.weights = np.zeros((20,350))
        self.sz  = np.zeros((20,350))

    # polyfit
    def f_3(self, x, A, B, C, D):
        return A * x * x * x + B * x * x + C * x + D

    def diff(self, x, A, B, C):
        dx = A * 3 * x * x + 2 * B * x + C
        return dx

    # get the curve fit for the original data
    def curvefit(self, x, y):
        const = dict()
        A, B, C, D = optimize.curve_fit(self.f_3, x, y)[0]
        if A != 0:
            const['A'] = A
        if B != 0:
            const['B'] = B
        if C!= 0:
            const['C'] = C
        if D!=0:
            const['D'] = D
        return const

    # calculate 2-Norm
    def distance(self, x,y,order):
       # print(norm(x - y,ord= order))
        return norm(x - y,ord= order)


    # 高斯函数
    def gaussian(self,pair, center, sigma):
        dis = self.distance(pair, center, order=2)
      #  print(dis,sigma)
        value = - self.distance(pair, center,order=2)
       # print(-(value*value )/ (sigma * sigma))
       # print(value*value,sigma*sigma)
       # print(exp(-(value*value )/ (sigma * sigma)))
        return exp(-(value*value )/ (sigma * sigma))


    def update_weights(self, gauss, eps, i, j, w):
        self.weights[i][j]=(- self.gamma * (gauss * eps + self.sigma * w))* 0.0005 + w
       # print(i,j,(- self.gamma * (gauss * eps + self.sigma * w))* 3 + w)
        return

    # calculating the nearest gaussian function around the trajectory
    # radius is the radius of sensitive field
    # RBF_center should be a list [[eps_min, eps_min+interval ..eps_max],[eps_min, eps_min+interval ..eps_max]].....
    # weights is a matrix data structure
    # a trajectory stands for one company
    def localized_core(self,trajectory, RBF_center, radius):
        x = np.arange(3,63,3)
        y = trajectory
        curve_parameters = self.curvefit(x,y)
        num = 3
        for points in trajectory:   # points from trajectory stored as a list
            for i in range(len(RBF_center)):   # searching every neuron centers
                for centers in RBF_center[i]:    #  vertical value
                    center = array([(i+1)*3,centers])    # make it the center a array to compute 2norm
                    point  = array([num,points])
                    if self.distance(center ,point,2)>=0:
                        j = RBF_center[i].index(centers)

                        self.sz[i][j] = self.distance(center ,point,2)
                        #print("distance 232")
                        gauss = self.gaussian(center,point,0.0001)
                        eps_ = self.diff(point[0],curve_parameters['A'],curve_parameters['B'],curve_parameters['C']) - point[1]
                        w = self.weights[i][j]
                        self.update_weights(gauss=gauss,eps=eps_,i=i,j=j,w=w)
            num = num +3

        return self.weights

    def output(self,weights):
        weight = array(weights)
        sz = array(self.sz)
        out0 = sz * weight
        return out0

    # classify shares into 5 groups
    # A : 0 B:3 C:1 D:4 E:2
    def classify(self,namelist):
        groups = dict()
        A = [0, 1, 3, 4, 8, 15, 17, 19, 23, 28, 29, 30, 34, 42, 43, 44, 46, 49]
        B = [2, 5, 7, 13, 14, 16, 18, 21, 25, 27, 31, 32, 35, 39, 40, 45]
        C = [11, 26, 37, 41]
        D = [36]
        E = [9, 10, 6, 12, 20, 22, 24, 33, 38, 48, 47]
        A1 = list()
        B1 = list()
        C1 = list()
        D1 = list()
        E1 = list()
        for i in A:
            A1.append(namelist[i])
        for i in B:
            B1.append(namelist[i])
        for i in C:
            C1.append(namelist[i])
        for i in D:
            D1.append(namelist[i])
        for i in E:
            E1.append(namelist[i])

        groups['A'] = A1
        groups['B'] = B1
        groups['C'] = C1
        groups['D'] = D1
        groups['E'] = E1
        return groups




    def predict(self,trajectory ,A_average,B_average,C_average,D_average,E_average):
        A = self.transfer(self.output(A_average),'purple')
        A= [x-0.21 for x in A]
        B = self.transfer(self.output(B_average),'yellow')
        B = [x-0.32 for x in B]
        C = self.transfer(self.output(C_average),'green')
        C = [x-0.73 for x in C]
        D = self.transfer(self.output(D_average),'red')
        D = [x-0.9 for x in D]
    #    print(self.output((E_average)))

        E = self.transfer(self.output(E_average),'orange')
       # E = [x - 2 for x in E]
       # print(A,B)
        x = np.linspace(3,60,20)
        y= list()

        for group in [A,B,C,D,E]:
            y.append(group)

        mininum = 0
        result = 0
        testdata = array([x,trajectory])
        color = ['red', 'yellow', 'green', 'black', 'pink']
        #plt.plot(x,trajectory,"blue")

        for i in range(5):
            #print(y[i])
            y1 =array(y[i])
            category = array([x,y[i]])
            #plt.plot(x,y[i],color[i])
            #print( self.distance(testdata,category,order=2))
            if mininum > self.distance(testdata,category,order=2) or mininum == 0:
                mininum = self.distance(testdata,category,order=2)
                result = i

        return result


    def average(self,x,y):
        i =0
        o =list()
        y0 =list()
        for i in np.linspace(3,60,20):
            lst=list()
            index = 0
            for elements in x:
                if elements == i:
                    lst.append(y[index])
                index = index + 1
            o.append(lst)

        #print(o)
        for data in o:
            if len(data) ==0:
                y0.append(0)
            else:
                y0.append(sum(data)/len(data))

        return y0

    def transfer(self,output,color):
        x0 = list()
        y0 = list()
        for i0 in range(20):
            for j0 in range(350):
                if output[i0][j0] != 0:
                    x0.append(i0)
                    y0.append(j0)

        m = 0
        while m < len(x0):
            x0[m] = (x0[m]+1) * 3
            y0[m] = float(y0[m]) * 0.1 - 10
            m = m + 1
       # print(x0)
       # print(y0)

        #print(self.average(x0,y0))
        y = self.average(x0,y0)
       # plt.plot(x0,y0,color)
       # parameters = self.curvefit(x0,y0)
        return y

#initialize RBF centers
volume =list()
i = -10
while i <= 25:
    volume.append(i)
    i = i + 0.1
#print(volume,len(volume))
j =0
RBF_centers=list()
while j <20:
    RBF_centers.append(volume)
    j = j + 1
#print(RBF_centers)
p = read_eps("E:\onedrive\graduation\database\epsvalue")
namelist = list()
for names in p.keys():
    namelist.append(names)

dlearning = d_learing()
groups = dlearning.classify(namelist)
#print(groups)
def main_function(trajectory,RBF_centers):
    radius = 0.02
    #dlearning.weights = np.zeros((20, 350))
    #dlearning.sz  = np.zeros((20,350))
    w = dlearning.localized_core(trajectory=trajectory,RBF_center=RBF_centers,radius=radius)
    return w

w1 = np.zeros((20,350))
#dlearning.weights = np.zeros((20,350))
#dlearning.sz  = np.zeros((20,350))
for stocks in groups['A']:
    w1 = main_function(p[stocks],RBF_centers) + w1
A_average = w1 / 16

w2 = np.zeros((20,350))
dlearning.weights = np.zeros((20,350))
dlearning.sz  = np.zeros((20,350))
for stocks in groups['B']:
    w2 = main_function(p[stocks],RBF_centers) + w2
B_average = w2 / 31

w3 = np.zeros((20,350))
dlearning.weights = np.zeros((20,350))
dlearning.sz  = np.zeros((20,350))
for stocks in groups['C']:
    w3 = main_function(p[stocks],RBF_centers) + w3
C_average = w3 / 1

w4 = np.zeros((20,350))
dlearning.weights = np.zeros((20,350))
dlearning.sz  = np.zeros((20,350))
for stocks in groups['D']:
    w4 = main_function(p[stocks],RBF_centers) + w4
D_average = w4 / 1


w5 = np.zeros((20,350))
dlearning.weights = np.zeros((20,350))
dlearning.sz  = np.zeros((20,350))
for stocks in groups['E']:
    w5 = main_function(p[stocks],RBF_centers) + w5
E_average = w5 / 1





print("hello")

for name in namelist:
   dlearning.weights = np.zeros((20,350))
   dlearning.sz  = np.zeros((20,350))
   x5 = np.linspace(3, 60, 20)
   y5 = p[name]
   plt.plot(x5,y5,"blue")
   wtest = main_function(p[name],RBF_centers)
   print(dlearning.predict(trajectory=p[name],A_average=A_average,B_average=B_average,C_average=C_average,D_average=D_average,E_average=E_average))

print("hello")
testdata = read_eps("E:\onedrive\graduation\database\\testeps")


for names0 in testdata.keys():
    dlearning.weights = np.zeros((20,350))
    dlearning.sz  = np.zeros((20,350))
    rtest = main_function(testdata[names0],RBF_centers)
    print(dlearning.predict(trajectory=testdata[names0],A_average=A_average,B_average=B_average,C_average=C_average,D_average=D_average,E_average=E_average))


#plt.show()





plt.show()
