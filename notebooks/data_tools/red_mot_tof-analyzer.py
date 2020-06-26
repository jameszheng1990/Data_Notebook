import os, h5py, re, time
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

exp_name = 'red_mot_TOF'

current_path = os.getcwd()

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


pixel_size = 10.5 # [um]
cross_section = 0.1014 # [um^2]
linewidth = 201.06192983 # [ns?]
pulse_length = 10 # [us]
efficiency = 0.50348 
gain = 1
high_intensity_coefficient = 2 / (linewidth * pulse_length * efficiency * gain)
low_intensity_coefficient = pixel_size**2 / cross_section

t_delay = 0.1 # in ms

def fit_func(x, x0, w, a, c):
    return a*np.exp(-pow(x - x0, 2)/(2*pow(w, 2))) + c

for i in folder_list:
    t = re.search('{}_(.*)#'.format(exp_name), i)
    t = float(t.group(1))
    t_list.append(t + t_delay) # in ms
    
    exp_path = os.path.join(current_path, i)
    exp_list = os.listdir(exp_path)
    hdf5_list = [j for j in exp_list if '.hdf5' in j]
    
    images = {}
    
    if hdf5_list:
        y_len = len(hdf5_list)
        for j in hdf5_list:
            
            t1=time.time()
            
            file_path = os.path.join(exp_path, j)
            with h5py.File(file_path, 'r') as images_h5:
                for key in images_h5:
                    images[key] = np.array(images_h5[key], dtype = 'float64')
                images_h5.close()
                
            image = np.array(images['image'], dtype='f')
            image = image[250:750, 650:950]
            bright = np.array(images['bright'], dtype='f')
            bright = bright[250:750, 650:950]
            
            X = range(image.shape[1])
            Y = range(image.shape[0])
            
            n = (
                low_intensity_coefficient * np.log(abs(bright+ 0.0001)/abs(image + 0.0001))
                + high_intensity_coefficient * (bright - image)
                )
            
            od = (
                np.log(abs(bright+ 0.01)/abs(image + 0.01))
                # + high_intensity_coefficient * (bright - image)
                )
            # od = (np.log((bright+0.1)/(image+0.1)))
            od = np.flipud(od)
            n = np.flipud(n)
            
            print(time.time()-t1)
            
            x_trace = np.sum(od, axis = 0)
            y_trace = np.sum(od, axis = 1)
                        
            p0x = (x_trace.argmax(), 60, np.max(x_trace), np.min(x_trace))
            p0y = (y_trace.argmax(), 45,  np.max(y_trace), np.min(y_trace))
            
            poptx, pcovx =  optimize.curve_fit(fit_func, X, x_trace, p0 = p0x)
            popty, pcovy =  optimize.curve_fit(fit_func, Y, y_trace, p0 = p0y)
            
            xcen = poptx[0]
            ycen = popty[0]
            xwidth = poptx[1]
            ywidth = popty[1]
            
            # plt.figure()
            # t_plot = np.linspace(0, np.max(X), 100)
            # plt.scatter(X, x_trace, c = 'blue')
            # plt.plot(t_plot, fit_func(t_plot, poptx[0], poptx[1], poptx[2], poptx[3]))
            # plt.figure()
            # t_plot = np.linspace(0, np.max(Y), 100)
            # plt.scatter(Y, y_trace, c = 'red')
            # plt.plot(t_plot, fit_func(t_plot, popty[0], popty[1], popty[2], popty[3]))
            
    n_list.append(n)            
    od_list.append(od)
    x_trace_list.append(x_trace)
    y_trace_list.append(y_trace)
    x_cen_list.append(xcen)
    y_cen_list.append(ycen)
    x_width_list.append(abs(xwidth))
    y_width_list.append(abs(ywidth))

# Plot OD image

plt.figure()
cmap = plt.get_cmap('jet')
plt.pcolormesh(X, Y, n_list[-1], cmap = cmap)
plt.colorbar()

# Fitted displacement in Y

plt.figure()

t_list = np.array(t_list)/1000
y_cen_list = np.array(y_cen_list)

def fit_free_fall(t, y0, eff_g):
    return y0 - 1/2*eff_g*pow(t, 2)

p0_ff = (300, 9.8e5)
popt_ff, pcov_ff = optimize.curve_fit(fit_free_fall, t_list, y_cen_list, p0 = p0_ff)
perr_ff = np.sqrt(np.diag(pcov_ff))

t_plot = np.linspace(0, np.max(t_list), 100)
plt.scatter(t_list, x_cen_list, label = 'X trace center', c = 'blue')
plt.scatter(t_list, y_cen_list, label = 'Y trace center', c = 'red')
plt.plot(t_plot, fit_free_fall(t_plot, popt_ff[0], popt_ff[1]))
plt.legend(loc ='best')

g = 9.85 # m/s^2
pixel_size = g/popt_ff[1]
print('Fitted pixel size is {} m.'.format(pixel_size))

# # TO verify g:

# plt.figure()

# y_cen_list2 = y_cen_list*pixel_size

# p0_ff2 = (300*pixel_size, 9.85)
# popt_ff2, pcov_ff2 = optimize.curve_fit(fit_free_fall, t_list, y_cen_list2, p0 = p0_ff2)
# plt.scatter(t_list, y_cen_list2, label = 'Y trace center', c = 'red')
# plt.plot(t_plot, fit_free_fall(t_plot, popt_ff2[0], popt_ff2[1]))
# plt.legend(loc ='best')

# fit_g = popt_ff2[1]
# print('Fitted g is {} m/s^2.'.format(fit_g))

## Fitted velocity, therefore temperature ##

def velocity(t, a, b):
    return np.sqrt(pow(a, 2) + pow(b*t, 2))

kb = 1.38e-23
m = 88* 1.672e-27
pixel_size = 10.29e-6

y_width_list = np.array(y_width_list)*pixel_size
x_width_list = np.array(x_width_list)*pixel_size

p0_v = (100, 8)

popt_vy, pcov_vy = optimize.curve_fit(velocity, t_list, y_width_list, p0 = p0_v)
perr_vy = np.sqrt(np.diag(pcov_vy))
popt_vx, pcov_vx = optimize.curve_fit(velocity, t_list, x_width_list, p0 = p0_v)
perr_vx = np.sqrt(np.diag(pcov_vx))

t_plot = np.linspace(0, np.max(t_list), 100)

plt.figure()
t_list= t_list*1e3
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

##############################################################################
    
def atom_number(od):
    pixel_size = 10.00e-6
    lambda_461 = 461e-9
    cross_section = 3*lambda_461**2/(2*np.pi)
    atom_num = np.sum(od)*pixel_size**2 / cross_section
    return atom_num
