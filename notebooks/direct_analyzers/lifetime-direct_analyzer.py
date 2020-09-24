import os, h5py, re, time, json
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from camera.clients.data_tools.process_image import process_images_g_fluorescence

def fit_Gaussian(x, x0, w, amp, const):
    return amp * np.exp(-pow(x - x0, 2)/(2*pow(w, 2))) + const

def fit_exp_decay(t, a, tau):
    return a* np.exp(-t/tau)

current_folder = os.getcwd()

folder_list = os.listdir()
ixon_list = [i for i in folder_list if '.ixon' in i]

t_list = []
count_list = []

data_folder = current_folder
roi_width = 25
fit_width = 10

xdata = []
ydata = []

for shot in range(len(ixon_list)):
    conductor_json_path = data_folder + '\\{}'.format(shot) + '.conductor.json'
    tcam_hdf5_path = data_folder + '\\{}'.format(shot) + '.ixon.hdf5'
    
    f = open(conductor_json_path, 'r')
    f1 = f.read()
    f2 = json.loads(f1)
    t = f2['sequencer.DO_parameters.lattice_hold'] 
    
    images = {}
    with h5py.File(tcam_hdf5_path, 'r') as images_h5:
        for key in images_h5:
            images[key] = np.array(images_h5[key], dtype = 'float64')
        images_h5.close()
            
    n0 = process_images_g_fluorescence(images)
    n = n0
            
    X = range(n.shape[1])
    Y = range(n.shape[0])
                
    x_trace = np.sum(n, axis = 0)
    y_trace = np.sum(n, axis = 1)
                        
    p0x = (x_trace.argmax(), fit_width, np.max(x_trace), np.min(x_trace))
    p0y = (y_trace.argmax(), fit_width, np.max(y_trace), np.min(y_trace))
            
    try:
        poptx, pcovx =  optimize.curve_fit(fit_Gaussian, X, x_trace, p0 = p0x)
        popty, pcovy =  optimize.curve_fit(fit_Gaussian, Y, y_trace, p0 = p0y)
            
    except:
        poptx = p0x
        popty = p0y
            
    count1 = np.sum(x_trace) - poptx[-1]*len(x_trace) # test
    count2 = np.sum(y_trace) - popty[-1]*len(y_trace) # test
    count = (count1 + count2)/2
    
    xdata.append(t)
    ydata.append(count)



p0 = [np.max(ydata), 0.7*np.max(xdata)]
    
try:
        popt, pcov = optimize.curve_fit(fit_exp_decay, xdata, ydata, p0 = p0)
        perr = np.sqrt(np.diag(pcov))
    
except Exception as e:
        print(e)
        popt = p0

t1 = np.arange(np.min(xdata) - 1, np.max(xdata)+ 1, 0.1)

f, ax = plt.subplots(1,1)
ax.scatter(xdata, ydata, c='red', marker = 's', s= 35)
ax.plot(t1, fit_exp_decay(t1, popt[0], popt[1]), label= '{}({}) s'.format(str(round(popt[1], 2)), str(round(perr[1], 2))),
             c='red', linestyle='dashed')
ax.set_title('Lattice lifetime measurement')
ax.set_yscale('log')
plt.legend(loc='best')