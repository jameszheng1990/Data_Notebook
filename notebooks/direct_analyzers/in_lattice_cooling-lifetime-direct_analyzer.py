import os, h5py, re, time
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import json

def fit_exp_decay(t, a, tau):
    return a* np.exp(-t/tau)

file_path1 = os.path.join(os.getcwd(), "lattice_lifetime-pump14W-ixon_withLock_-0.45_withCooling-79.770MHz#0", "result\\processed_data.txt")
file_path2 = os.path.join(os.getcwd(), "lattice_lifetime-pump14W-ixon_withLock_-0.45_withoutCooling-79.770MHz#0", "result\\processed_data.txt")

data1 = {}
with open(file_path1, 'r') as f:
    f1 = f.read()
    data1 = json.loads(f1)

data2 = {}
with open(file_path2, 'r') as f:
    f1 = f.read()
    data2 = json.loads(f1)

data1 = sorted(data1, key = lambda k:k['shot'])
data2 = sorted(data2, key = lambda k:k['shot'])

x_data1 = [i["lattice_hold"] for i in data1 ]
x_data2 = [i["lattice_hold"] for i in data2 ]

y_data1 = [i["fit"]['ccd_count'] for i in data1 ]
y_data2 = [i["fit"]['ccd_count'] for i in data2 ]

y_data1 = [i/max(y_data1) for i in y_data1]
y_data2 = [i/max(y_data2) for i in y_data2]

p1 = [max(y_data1), 0.7*max(y_data1)]
p2 = [max(y_data2), 0.7*max(y_data2)]
    
try:
   popt1, pcov1 = optimize.curve_fit(fit_exp_decay, x_data1, y_data1, p0 = p1)
   perr1 = np.sqrt(np.diag(pcov1))
    
except Exception as e:
   print(e)
   popt1 = p1
   
try:
   popt2, pcov2 = optimize.curve_fit(fit_exp_decay, x_data2, y_data2, p0 = p2)
   perr2 = np.sqrt(np.diag(pcov2))
    
except Exception as e:
   print(e)
   popt2 = p2

t1 = np.arange(0, max(x_data1) +5, 0.1)
t2 = np.arange(0, max(x_data2) +5, 0.1)

plt.figure()
plt.scatter(x_data1, y_data1, c = 'r')
plt.plot(t1, fit_exp_decay(t1, popt1[0], popt1[1]), c = 'r', linestyle='dashed',
         label = 'with cooling, {}({}) s'.format(round(popt1[1], 2), round(perr1[1], 2)))
plt.scatter(x_data2, y_data2, c = 'blue')
plt.plot(t2, fit_exp_decay(t2, popt2[0], popt2[1]), c = 'blue', linestyle='dashed',
         label = 'without cooling, {}({}) s'.format(round(popt2[1], 2), round(perr2[1], 2)))
plt.xlabel('lattice hold [s]')
plt.ylabel('Normalized atom number')
plt.legend()
plt.show()

plt.savefig('In_lattice_cooling-lifetime_comapre.png')   