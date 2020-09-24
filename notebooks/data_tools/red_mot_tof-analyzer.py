import os, h5py, re, time, json, time
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from camera.clients.data_tools.process_image import process_images_g
from camera.clients.data_tools.process_image import get_pixel_size_absorption

import sys
sys.path.insert(1, 'C:\\LabRad\\SrData\\data\\notebooks\\data_tools')
from helper import process_units

PROJECT_DATA_PATH = os.path.join(os.getenv('LABRADDATA'), 'data')

exp_name = 'red_mot_tof'

t_delay = 0.1e-3
kb = 1.38e-23
m = 88* 1.672e-27
# pixel_size = get_pixel_size_absorption()*1e-6

def fit_Gaussian(x, x0, w, amp, const):
    return amp * np.exp(-pow(x - x0, 2)/(2*pow(w, 2))) + const

def fit_free_fall(t, y0, eff_g):
    return y0 - 1/2*eff_g*pow(t, 2)

def fit_velocity(t, w0, v):
    return np.sqrt(pow(w0, 2) + pow(v*t, 2))

def process_red_mot_tof_ccd(settings):
    shot = settings['shot']
    roi_width = settings['kwargs']['roi_width']
    fit_width = settings['kwargs']['fit_width']
    x_key = settings['kwargs']['x_key']
    y_key = settings['kwargs']['y_key']
    method = settings['kwargs']['method']
    
    if shot >= 0:
        data = []
        data_folder = os.path.join(PROJECT_DATA_PATH, settings['data_path'])
        save_folder = os.path.join(PROJECT_DATA_PATH, settings['data_path'], 'processed_data')
        save_path = os.path.join(save_folder, str(shot))
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
            
        conductor_json_path = data_folder + '\\{}'.format(shot) + '.conductor.json'
        ccd_hdf5_path = data_folder + '\\{}'.format(shot) + '.{}.hdf5'.format(method)
        
        try:
            f = open(conductor_json_path, 'r')
            f1 = f.read()
            f2 = json.loads(f1)
            tof = f2['sequencer.DO_parameters.red_mot_tof'] + t_delay
            try:
                em_gain = f2['andor.record_path']['em_gain']
            except:
                em_gain = 1
            
            images = {}
            with h5py.File(ccd_hdf5_path, 'r') as images_h5:
                for key in images_h5:
                    images[key] = np.array(images_h5[key], dtype = 'float64')
                images_h5.close()
        
            n0 = process_images_g(images, em_gain)
            x_trace0 = np.sum(n0, axis = 0)
            y_trace0 = np.sum(n0, axis = 1)
            xc = x_trace0.argmax()
            yc = y_trace0.argmax()
            n = n0
            n = np.flipud(n)
            
            X = range(n.shape[1])
            Y = range(n.shape[0])
            
            x_trace = np.sum(n, axis = 0)
            y_trace = np.sum(n, axis = 1)
                            
            p0x = (x_trace.argmax(), fit_width, np.max(x_trace), np.min(x_trace))
            p0y = (y_trace.argmax(), fit_width, np.max(y_trace), np.min(y_trace))
                
            try:
                poptx, pcovx =  optimize.curve_fit(fit_Gaussian, X, x_trace, p0 = p0x)
                popty, pcovy =  optimize.curve_fit(fit_Gaussian, Y, y_trace, p0 = p0y)
        
            except Exception as e1:
                print(e1)
                poptx = p0x
                popty = p0y
            
            xcen = poptx[0]
            ycen = popty[0]
            xwidth = poptx[1]
            ywidth = popty[1]
            
            count1 = np.sum(x_trace) - poptx[-1]*len(x_trace) # test
            count2 = np.sum(y_trace) - popty[-1]*len(y_trace) # test
            count = (count1+count2)/2
            count = round(count, 1)
            
            data.append({'shot':shot, x_key: tof,  'fit': {'xcen':xcen, 'ycen':ycen, 'xwidth':xwidth, 'ywidth': ywidth, 'count': count}})
            
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(20, 8)
            
            n1 = n0[yc - roi_width: yc + roi_width, xc - roi_width: xc + roi_width]
            X = range(n1.shape[1])
            Y = range(n1.shape[0])
            
            cmap = plt.get_cmap('jet')
            ax.set_aspect('equal')
            plt.pcolormesh(X, Y, n1, cmap = cmap, vmin = 0)
            plt.colorbar()
            plt.title('Atom number: {:.2e}'.format(count))
                
            plt.savefig(save_path + '.png', bbox_inches='tight')
            
            
            X = range(n.shape[1])
            Y = range(n.shape[0])
            plt.figure()
            t_plot = np.linspace(np.min(X), np.max(X), 100)
            plt.scatter(X, x_trace, c = 'blue')
            plt.plot(t_plot, fit_Gaussian(t_plot, poptx[0], poptx[1], poptx[2], poptx[3]))
            plt.savefig(save_path + '_traceX.png', bbox_inches='tight')
            plt.figure()
            t_plot = np.linspace(np.min(Y), np.max(Y), 100)
            plt.scatter(Y, y_trace, c = 'red')
            plt.plot(t_plot, fit_Gaussian(t_plot, popty[0], popty[1], popty[2], popty[3]))
            plt.savefig(save_path + '_traceY.png', bbox_inches='tight')
            
            plt.clf()
            plt.close('all')
            
        except Exception as e:
            print(e)
            data = []
            xcen = 0
            ycen = 0
            xwidth = 0
            ywidth = 0
            count = 0
            data.append({'shot':shot, x_key: tof, 'fit': {'xcen':xcen, 'ycen':ycen, 'xwidth':xwidth, 'ywidth': ywidth, 'count': count}})
            
        return  data
    
    else:
        return data

def plot_red_mot_tof_ccd(data, settings):
    units = settings['kwargs']['units']
    data_range = settings['kwargs']['data_range']
    x_label = settings['kwargs']['x_label']
    y_label = settings['kwargs']['y_label']
    x_key = settings['kwargs']['x_key']
    y_key = settings['kwargs']['y_key']
    
    result_folder = os.path.join(PROJECT_DATA_PATH, settings['data_path'], 'result')
    result_path = os.path.join(result_folder, os.path.split(settings['data_path'])[0] + '_' + os.path.split(settings['data_path'])[1])
    saved_data_path = os.path.join(result_folder, 'processed_data.txt')
    
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)
    
    # sort by shot number
    sorted_data = sorted(data, key = lambda k:k['shot'])
    x_data = [i[x_key] for i in sorted_data if min(data_range) <= i['fit']['count'] < max(data_range)]
    y_data1 = [i['fit']['ycen'] for i in sorted_data if min(data_range) <= i['fit']['count'] < max(data_range)]
    y_data2 = [i['fit']['xwidth'] for i in sorted_data if min(data_range) <= i['fit']['count'] < max(data_range)]
    y_data3 = [i['fit']['ywidth'] for i in sorted_data if min(data_range) <= i['fit']['count'] < max(data_range)]
    
    with open(saved_data_path, 'w') as file:
        file.write(json.dumps(sorted_data))
    
    x_data = np.array(x_data)
    t_plot = np.linspace(0, 1e3*np.max(x_data), 100)
    
    # Free fall fit, calibrates the CCD pixel
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(20, 8)
    
    p0_ff = (max(y_data1), 9.8e5)
    popt_ff, pcov_ff = optimize.curve_fit(fit_free_fall, x_data, y_data1, p0 = p0_ff)
    perr_ff = np.sqrt(np.diag(pcov_ff))
    
    ax.scatter(1e3*x_data, y_data1, label = 'Y trace center', c = 'red')
    ax.plot(t_plot, fit_free_fall(1e-3*t_plot, popt_ff[0], popt_ff[1]))
    ax.legend(loc ='best')
    ax.set_ylabel('Fitted center')
    ax.set_xlabel(x_label + ' [{}]'.format(units))
    # max_point = max([max(l.get_ydata()) for l in ax.get_lines()])
    # min_point = min([min(l.get_ydata()) for l in ax.get_lines()])
    
    # ax[0].set_ylim([min(0, 0.8*min_point), max_point*1.2])
    
    g = 9.85 # m/s^2
    pixel_size = g/popt_ff[1]
    plt.title(r'Red mot tof, free fall fit, pixel size = {} um'.format(round(pixel_size*1e6, 2)))
    plt.savefig(result_path+'_free_fall_fit.svg', bbox_inches='tight')
    
    plt.clf()
    plt.close('all')
    
    # TOF fit, gets the temperature
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(20, 8)
    
    x_width_list = np.array(y_data2)*pixel_size 
    y_width_list = np.array(y_data3)*pixel_size   
    
    p0_v = (100, 5)
    popt_vx, pcov_vx = optimize.curve_fit(fit_velocity, x_data, x_width_list, p0 = p0_v)
    perr_vx = np.sqrt(np.diag(pcov_vx))
    popt_vy, pcov_vy = optimize.curve_fit(fit_velocity, x_data, y_width_list, p0 = p0_v)
    perr_vy = np.sqrt(np.diag(pcov_vy))
    
    # print(x_data, popt_vx)
    
    temp_x = m*(popt_vx[1])**2/kb
    temp_y = m*(popt_vy[1])**2/kb
    
    ax.scatter(1e3*x_data, 1e6*x_width_list, label = r'T$_x$ = {} $\mu$K'.format(round(temp_x*1e6, 2)), c = 'blue')
    ax.scatter(1e3*x_data, 1e6*y_width_list, label = r'T$_y$ = {} $\mu$K'.format(round(temp_y*1e6, 2)), c = 'red')
    ax.plot(t_plot, 1e6*fit_velocity(1e-3*t_plot, popt_vx[0], popt_vx[1]), c ='blue')
    ax.plot(t_plot, 1e6*fit_velocity(1e-3*t_plot, popt_vy[0], popt_vy[1]), c= 'red')
    ax.set_ylabel(r'1/e$^2$ cloud width ($\mu$m)')
    ax.set_xlabel(x_label + ' [{}]'.format(units))
    # max_point = max([max(l.get_ydata()) for l in ax.get_lines()])
    # min_point = min([min(l.get_ydata()) for l in ax.get_lines()])
    
    # ax[0].set_ylim([min(0, 0.8*min_point), max_point*1.2])
    ax.legend(loc ='best')
    plt.title('Red mot tof temperature')
    plt.savefig(result_path+'.svg', bbox_inches='tight')
    
    plt.clf()
    plt.close('all')
    # plt.show()

