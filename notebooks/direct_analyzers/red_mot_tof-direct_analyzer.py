import os, h5py, re, time, json
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from camera.clients.data_tools.process_image import process_images_g

exp_name = 'red_mot_tof'

current_folder = os.getcwd()

folder_list = os.listdir()
folder_list = [i for i in folder_list if '#' in i and '.py' not in i]

t_list = []
n_list = []
od_list = []
x_trace_list = []
y_trace_list = []
x_cen_list = []
y_cen_list = []
x_width_list = []
y_width_list = []


t_delay = 0.1e-3 # in ms


def fit_Gaussian(x, x0, w, a, c):
    return a*np.exp(-pow(x - x0, 2)/(2*pow(w, 2))) + c

data_folder = current_folder
roi_width = 40
fit_width = 10

xdata = []
ydata = []

for shot in range(5):
    conductor_json_path = data_folder + '\\{}'.format(shot) + '.conductor.json'
    tcam_hdf5_path = data_folder + '\\{}'.format(shot) + '.ixon.hdf5'
    
    f = open(conductor_json_path, 'r')
    f1 = f.read()
    f2 = json.loads(f1)
    t = f2['sequencer.DO_parameters.red_mot_tof'] + t_delay
    
    images = {}
    with h5py.File(tcam_hdf5_path, 'r') as images_h5:
        for key in images_h5:
            images[key] = np.array(images_h5[key], dtype = 'float64')
        images_h5.close()
            
    n0 = process_images_g(images)
    n = n0
    # yc, xc = np.unravel_index(n0.argmax(), n0.shape)
    # n = n0[yc - roi_width: yc + roi_width, xc - roi_width: xc + roi_width]
    n = np.flipud(n)
    # print(yc, xc)
            
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
    
    xcen = poptx[0]
    ycen = popty[0]
    xwidth = poptx[1]
    ywidth = popty[1]
    
    
    x_trace_list.append(x_trace)
    y_trace_list.append(y_trace)
    x_cen_list.append(xcen)
    y_cen_list.append(ycen)
    x_width_list.append(abs(xwidth))
    y_width_list.append(abs(ywidth))
            
    xdata.append(t)
    ydata.append(count)
    
    plt.figure()
    t_plot = np.linspace(np.min(X), np.max(X), 100)
    plt.scatter(X, x_trace, c = 'blue')
    plt.plot(t_plot, fit_Gaussian(t_plot, poptx[0], poptx[1], poptx[2], poptx[3]))
    plt.figure()
    t_plot = np.linspace(np.min(Y), np.max(Y), 100)
    plt.scatter(Y, y_trace, c = 'red')
    plt.plot(t_plot, fit_Gaussian(t_plot, popty[0], popty[1], popty[2], popty[3]))

    plt.figure()
    cmap = plt.get_cmap('jet')
    plt.pcolormesh(X, Y, n, cmap = cmap)
    plt.colorbar()

def fit_free_fall(t, y0, eff_g):
    return y0 - 1/2*eff_g*pow(t, 2)

t_list = xdata

p0_ff = (300, 9.8e5)
popt_ff, pcov_ff = optimize.curve_fit(fit_free_fall, t_list, y_cen_list, p0 = p0_ff)
perr_ff = np.sqrt(np.diag(pcov_ff))

plt.figure()
t_plot = np.linspace(1e-3, np.max(t_list), 100)
plt.scatter(t_list, x_cen_list, label = 'X trace center', c = 'blue')
plt.scatter(t_list, y_cen_list, label = 'Y trace center', c = 'red')
plt.plot(t_plot, fit_free_fall(t_plot, popt_ff[0], popt_ff[1]))
plt.legend(loc ='best')

g = 9.85 # m/s^2
pixel_size = g/popt_ff[1]
print('Fitted pixel size is {} m.'.format(pixel_size))


## Fitted velocity, therefore temperature ##

def velocity(t, a, b):
    return np.sqrt(pow(a, 2) + pow(b*t, 2))

kb = 1.38e-23
m = 88* 1.672e-27

y_width_list = np.array(y_width_list)*pixel_size
x_width_list = np.array(x_width_list)*pixel_size

p0_v = (100, 8)

popt_vy, pcov_vy = optimize.curve_fit(velocity, t_list, y_width_list, p0 = p0_v)
perr_vy = np.sqrt(np.diag(pcov_vy))
popt_vx, pcov_vx = optimize.curve_fit(velocity, t_list, x_width_list, p0 = p0_v)
perr_vx = np.sqrt(np.diag(pcov_vx))

t_list = [i*1e3 for i in t_list]
t_plot = np.linspace(0, np.max(t_list), 100)

plt.figure()
y_width_list = y_width_list*1e6
x_width_list = x_width_list*1e6
t_plot = np.linspace(0, np.max(t_list), 100)

temp_y = m*(popt_vy[1])**2/kb
temp_x = m*(popt_vx[1])**2/kb

plt.scatter(t_list, x_width_list, label = r'T$_x$ = {} $\mu$K'.format(round(temp_x*1e6, 2)), c = 'blue')
plt.scatter(t_list, y_width_list, label = r'T$_y$ = {} $\mu$K'.format(round(temp_y*1e6, 2)) , c = 'red')
plt.plot(t_plot, 1e6*velocity(t_plot*1e-3, popt_vy[0], popt_vy[1]), c= 'red')
plt.plot(t_plot, 1e6*velocity(t_plot*1e-3, popt_vx[0], popt_vx[1]), c ='blue')
plt.xlabel('Time (ms)')
plt.ylabel(r'1/e$^2$ cloud width ($\mu$m)')
plt.legend(loc ='best')
print('Fitted y temperature is {} uK, x temperature is {} uK.'.format(round(temp_y*1e6, 2), round(temp_x*1e6, 2)))

plt.savefig('red_mot_temp.svg', bbox_inches='tight')
