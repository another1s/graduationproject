import csv
from readexcel import share_data
import os
import numpy as np
from numpy import array, append, vstack, transpose, reshape,dot, true_divide, mean, exp, sqrt, log,loadtxt, savetxt, zeros, frombuffer
from numpy.linalg import norm, lstsq
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import optimize

y0=list()
l =0.003
ts= 0.00006665
x = [10,100,1000,10000,100000,1000000,10000000]
for e in x:
    s=l*e*(1+ts*e)
    y=1/s
    y0.append(y)
x0 =np.arange(10,17,1)
x1 =['10','100','1000','10000','100000','1000000','10000000']
plt.plot(x1,y0,'red')

plt.show()
