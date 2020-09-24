import os, h5py, re, time
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import json

file_path1 = os.path.join(os.getcwd(), "lattice_trap_freq_modulation-Sr88-pump14W-holdtime0.25s-ixon#18", "result\\processed_data.txt")
file_path2 = os.path.join(os.getcwd(), "lattice_trap_freq_modulation-Sr88-pump14W-holdtime0.25s-ixon#19", "result\\processed_data.txt")

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

x_data1 = [i["modulation_frequency"]/1e3 for i in data1 ]
x_data2 = [i["modulation_frequency"]/1e3 for i in data2 ]

y_data1 = [i["fit"]['ccd_count'] for i in data1 ]
y_data2 = [i["fit"]['ccd_count'] for i in data2 ]

y_data1 = [i/max(y_data1) for i in y_data1]
y_data2 = [i/max(y_data2) for i in y_data2]

plt.figure()
plt.scatter(x_data1, y_data1, c = 'r', label = 'with cooling')
plt.plot(x_data1, y_data1, c = 'r')
plt.scatter(x_data2, y_data2, c = 'blue', label = 'without cooling')
plt.plot(x_data2, y_data2, c = 'blue')
plt.xlabel('Modulation frequency [kHz]')
plt.ylabel('Normalized atom number')
plt.ylim(0, 1)
plt.legend()
plt.show()

plt.savefig('In_lattice_cooling-trap_depth-compare.png')   