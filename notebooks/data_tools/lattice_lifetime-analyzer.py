import os, h5py, re, time, json
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from camera.clients.data_tools.process_image import process_images_g
from camera.clients.data_tools.process_image import process_images_g_fluorescence

import sys
sys.path.insert(1, 'C:\\LabRad\\SrData\\data\\notebooks\\data_tools')
from helper import process_units
# from data.notebooks.helper.data_tools import process_camera

PROJECT_DATA_PATH = os.path.join(os.getenv('LABRADDATA'), 'data')

exp_name = 'lattice_lifetime'
    
def fit_Gaussian(x, x0, w, amp, const):
    return amp * np.exp(-pow(x - x0, 2)/(2*pow(w, 2))) + const

def fit_exp_decay(t, a, tau):
    return a* np.exp(-t/tau)

def process_lattice_lifetime_ccd(settings):
    shot = settings['shot']
    roi_center = settings['kwargs']['roi_center']
    roi_width = settings['kwargs']['roi_width']
    fit_width = settings['kwargs']['fit_width']
    x_key = settings['kwargs']['x_key']
    y_key = settings['kwargs']['y_key']
    method = settings['kwargs']['method']
    image = settings['kwargs']['image']
    
    if shot >= 0:
        data = []
        data_folder = os.path.join(PROJECT_DATA_PATH, settings['data_path'])
        save_folder = os.path.join(PROJECT_DATA_PATH, settings['data_path'], 'processed_data')
        save_path = os.path.join(save_folder, str(shot))
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
            
        conductor_json_path = data_folder + '\\{}'.format(shot) + '.conductor.json'
        ccd_hdf5_path = data_folder + '\\{}'.format(shot) + '.{}.hdf5'.format(method)
                        
        with open(conductor_json_path, 'r') as file:
            f1 = file.read()
            f2 = json.loads(f1)
        try:
            em_gain = f2['andor.record_path']['em_gain']
        except:
            em_gain = 1
            
        try: 
            
            f = open(conductor_json_path, 'r')
            f1 = f.read()
            f2 = json.loads(f1)
            holdtime = f2['sequencer.DO_parameters.lattice_hold']
            
            images = {}
            with h5py.File(ccd_hdf5_path, 'r') as images_h5:
                for key in images_h5:
                    images[key] = np.array(images_h5[key], dtype = 'float64')
                images_h5.close()
                
            if image == 'absorption':
                n0 = process_images_g(images, em_gain)
                
                xc = n0.shape[1] -roi_center[1]
                yc = roi_center[0]
                n = n0[yc - roi_width: yc + roi_width, xc - roi_width: xc + roi_width]
                x_trace0 = np.sum(n, axis = 0)
                y_trace0 = np.sum(n, axis = 1)
                X = range(n.shape[1])
                Y = range(n.shape[0])
                count = np.sum(n)
                
                count = round(float(count), 1)
                    
            elif image == 'fluorescence':
                n0 = process_images_g_fluorescence(images)
                
                xc = n0.shape[1] -roi_center[1]
                yc = roi_center[0]
                n = n0[yc - roi_width: yc + roi_width, xc - roi_width: xc + roi_width]
                x_trace0 = np.sum(n, axis = 0)
                y_trace0 = np.sum(n, axis = 1)
                X = range(n.shape[1])
                Y = range(n.shape[0])
                count = np.sum(n)
                
                # fit Gaussian
                # try:
                #     x_trace0 = np.sum(n0, axis = 0)
                #     y_trace0 = np.sum(n0, axis = 1)
                #     X = range(n0.shape[1])
                #     Y = range(n0.shape[0])
                #     p0x = (x_trace0.argmax(), fit_width, np.max(x_trace0), np.min(x_trace0))
                #     p0y = (y_trace0.argmax(), fit_width, np.max(y_trace0), np.min(y_trace0))
                
                #     poptx, pcovx =  optimize.curve_fit(fit_Gaussian, X, x_trace0, p0 = p0x)
                #     popty, pcovy =  optimize.curve_fit(fit_Gaussian, Y, y_trace0, p0 = p0y)
                    
                #     countx = np.sum(x_trace0) - poptx[-1]*len(x_trace0)
                #     county = np.sum(y_trace0) - poptx[-1]*len(y_trace0)
                #     count = (countx+county)/2
                    
                # # sum - bg
                # except Exception as e:
                #     print(e)
                #     (xc, yc) = np.unravel_index(np.argmax(n0, axis=None), n0.shape)
                #     n = n0[xc - roi_width: xc + roi_width, yc - roi_width: yc + roi_width]
                
                #     n_bg1 = n0[0: roi_width, 0:roi_width]
                #     n_bg2 = np.mean(n_bg1)
                
                #     count = np.sum(n) - n_bg2*np.size(n)
                
                count = round(float(count), 1)
            
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(20, 8)
            
            cmap = plt.get_cmap('jet')
            ax.set_aspect('equal')
            plt.pcolormesh(X, Y, n, cmap = cmap)
            ax.set_xlim(0, n.shape[1])
            ax.set_ylim(0, n.shape[0])
            # plt.legend()
            plt.colorbar()
            plt.title('Atom number: {:.2e}'.format(count))
            
            plt.savefig(save_path + '.png', bbox_inches='tight')
            
            plt.clf()
            plt.close('all')
        
        except Exception as e2:
            print(e2)
            count = 0
        
        data.append({'shot': shot, x_key: holdtime, 'fit': {y_key: count}})
        return  data

    else:
        return data

def plot_lattice_lifetime_ccd(data, settings):
    # print(data)

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
    x_data = [i[x_key] for i in sorted_data]
    y_data = [i['fit'][y_key] for i in sorted_data]
    
    with open(saved_data_path, 'w') as file:
        file.write(json.dumps(sorted_data))
    
    x_data = process_units(x_data, units)
    
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    # y_data = y_data/(np.max(y_data)) # Normalize
    
    p0 = [np.max(y_data), 0.7*np.max(x_data)]
    
    try:
        popt, pcov = optimize.curve_fit(fit_exp_decay, x_data, y_data, p0 = p0)
        perr = np.sqrt(np.diag(pcov))
    
    except Exception as e:
        print(e)
        popt = p0

    t1 = np.arange(np.min(x_data) - 1, np.max(x_data)+ 1, 0.1)

    f, ax = plt.subplots(1,1)
    ax.scatter(x_data, y_data, c='red', marker = 's', s= 35)
    ax.plot(t1, fit_exp_decay(t1, popt[0], popt[1]), label= '{}({}) s'.format(str(round(popt[1], 2)), str(round(perr[1], 2))),
             c='red', linestyle='dashed')
    ax.set_xlabel(x_label+" ({})".format(units))
    ax.set_ylabel(y_label)
    ax.set_title('Lattice lifetime measurement')
    ax.set_yscale('log')
    plt.legend(loc='best')
    
    plt.savefig(result_path+'.svg', bbox_inches='tight')
    plt.savefig(result_path+'.png', bbox_inches='tight')
    
    plt.clf()
    plt.close('all')
    # plt.show()
    
def process_lattice_lifetime_pmt(settings):
    shot = settings['shot']
    x_key = settings['kwargs']['x_key']
    y_key = settings['kwargs']['y_key']
    count_key = settings['kwargs']['count_key']
    
    if shot >= 0:
        data = []
        data_folder = os.path.join(PROJECT_DATA_PATH, settings['data_path'])
        save_folder = os.path.join(PROJECT_DATA_PATH, settings['data_path'], 'processed_data')
        save_path = os.path.join(save_folder, str(shot))
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
            
        conductor_json_path = data_folder + '\\{}'.format(shot) + '.conductor.json'
        blue_pmt_path = data_folder + '\\{}'.format(shot) + '.blue_pmt.json'

        f = open(conductor_json_path, 'r')
        f1 = f.read()
        f2 = json.loads(f1)
        holdtime = f2['sequencer.DO_parameters.lattice_hold']
            
        f = open(blue_pmt_path, 'r')
        f1 = f.read()
        f2 = json.loads(f1)
        count = f2[count_key]
        
        data.append({'shot': shot, x_key: holdtime, y_key: count})
        return  data
        
    else:
        return data

def plot_lattice_lifetime_pmt(data, settings):

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
    x_data = [i[x_key] for i in sorted_data if min(data_range) <i[y_key] < max(data_range)]
    y_data = [i[y_key] for i in sorted_data if min(data_range) <i[y_key] < max(data_range)]
    
    with open(saved_data_path, 'w') as file:
        file.write(json.dumps(sorted_data))
    
    x_data = process_units(x_data, units)
    
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    y_data = y_data/(np.max(y_data))
    
    p0 = [np.max(y_data), 0.7*np.max(y_data)]
    
    try:
        popt, pcov = optimize.curve_fit(fit_exp_decay, x_data, y_data, p0 = p0)
        perr = np.sqrt(np.diag(pcov))
    
    except Exception as e:
        print(e)
        popt = p0

    # print(popt, perr)

    # lifetime = round(popt[1], 1)
    # errorbar = round(perr[1], 1)

    t1 = np.arange(np.min(x_data), np.max(x_data), 0.1)

    plt.figure()
    plt.ylabel(y_label)
    plt.xlabel(x_label+" ({})".format(units))
    plt.title('Lattice lifetime measurement')
    plt.scatter(x_data, y_data, c='red', marker = 's', s= 35)
    plt.plot(t1, fit_exp_decay(t1, popt[0], popt[1]), label= str(round(popt[1], 2 ))+'s' , c='red',
             linestyle='dashed')
    plt.legend(loc='best')
    
    plt.savefig(result_path+'.svg', bbox_inches='tight')
    
    plt.clf()
    plt.close('all')
    # plt.show()