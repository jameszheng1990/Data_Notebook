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

exp_name = 'lattice_trap_freq_modulation'
    
def fit_Gaussian(x, x0, w, amp, const):
    return amp * np.exp(-pow(x - x0, 2)/(2*pow(w, 2))) + const

def process_lattice_trap_freq_modulation_ccd(settings):
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
        tcam_hdf5_path = data_folder + '\\{}'.format(shot) + '.{}.hdf5'.format(method)
        
        try:
            f = open(conductor_json_path, 'r')
            f1 = f.read()
            f2 = json.loads(f1)
            freq = f2['lattice.ModulationFrequency']
            try:
                em_gain = f2['andor.record_path']['em_gain']
            except:
                em_gain = 1
            
            images = {}
            with h5py.File(tcam_hdf5_path, 'r') as images_h5:
                for key in images_h5:
                    images[key] = np.array(images_h5[key], dtype = 'float64')
                images_h5.close()
        
            if image == 'absorption':
                n0 = process_images_g(images, em_gain)
                
                xc = roi_center[0]
                yc = n0.shape[1] -roi_center[1]
                n = n0[xc - roi_width: xc + roi_width, yc - roi_width: yc + roi_width]
                x_trace0 = np.sum(n, axis = 0)
                y_trace0 = np.sum(n, axis = 1)
                X = range(n.shape[1])
                Y = range(n.shape[0])
                count = np.sum(n)
                
                count = round(float(count), 1)
                    
            elif image == 'fluorescence':
                n0 = process_images_g_fluorescence(images)
                
                xc = roi_center[0]
                yc = n0.shape[1] -roi_center[1]
                n = n0[xc - roi_width: xc + roi_width, yc - roi_width: yc + roi_width]
                x_trace0 = np.sum(n, axis = 0)
                y_trace0 = np.sum(n, axis = 1)
                X = range(n.shape[1])
                Y = range(n.shape[0])
                count = np.sum(n)
                
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
        
        except Exception as e:
            print(e)
            count = 0
            
        data.append({'shot': shot, x_key:freq, 'fit': {y_key: count}})
        return  data
    
            
    else:
        return data

def plot_lattice_trap_freq_modulation_ccd(data, settings):
    # print(data)

    units = settings['kwargs']['units']
    data_range = settings['kwargs']['data_range']
    x_label = settings['kwargs']['x_label']
    y_label = settings['kwargs']['y_label']
    x_key = settings['kwargs']['x_key']
    y_key = settings['kwargs']['y_key']
    sorted_key = settings['kwargs']['sorted_key']
    
    result_folder = os.path.join(PROJECT_DATA_PATH, settings['data_path'], 'result')
    result_path = os.path.join(result_folder, os.path.split(settings['data_path'])[0] + '_' + os.path.split(settings['data_path'])[1])
    saved_data_path = os.path.join(result_folder, 'processed_data.txt')
    
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)
    
    # sort by x_key
    sorted_data = sorted(data, key = lambda k:k[sorted_key])
    x_data = [i[x_key] for i in sorted_data ]
    y_data = [i['fit'][y_key] for i in sorted_data ]
    
    with open(saved_data_path, 'w') as file:
        file.write(json.dumps(sorted_data))
    
    x_data = process_units(x_data, units)
    
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(20, 8)
        
    ax.plot(x_data, y_data)
    ax.plot(x_data, y_data, 'ro')
    
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label + ' ({})'.format(units))
    # max_point = max([max(l.get_ydata()) for l in ax.get_lines()])
    # min_point = min([min(l.get_ydata()) for l in ax.get_lines()])
    
    # ax[0].set_ylim([min(0, 0.8*min_point), max_point*1.2])
    plt.title('Lattice trapping frequency measurement')
    plt.savefig(result_path+'.png', bbox_inches='tight')
    plt.savefig(result_path+'.svg', bbox_inches='tight')
    
    plt.clf()
    plt.close('all')
    # plt.show()

def process_lattice_trap_freq_modulation_pmt(settings):
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
        freq = f2['lattice.ModulationFrequency']
            
        f = open(blue_pmt_path, 'r')
        f1 = f.read()
        f2 = json.loads(f1)
        count = f2[count_key]
        
        data.append({'shot': shot, x_key: freq, y_key: count})
        return  data
    
    else:
        return data

def plot_lattice_trap_freq_modulation_pmt(data, settings):
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
    x_data = [i[x_key] for i in sorted_data if min(data_range) < i[y_key] < max(data_range) ]
    y_data = [i[y_key] for i in sorted_data if min(data_range) < i[y_key] < max(data_range)]
    
    with open(saved_data_path, 'w') as file:
        file.write(json.dumps(sorted_data))
    
    x_data = process_units(x_data, units)
    print(x_data, units)
    
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(20, 8)
        
    ax.plot(x_data, y_data)
    ax.plot(x_data, y_data, 'ro')
    
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label + ' ({})'.format(units))
    # max_point = max([max(l.get_ydata()) for l in ax.get_lines()])
    # min_point = min([min(l.get_ydata()) for l in ax.get_lines()])
    
    # ax[0].set_ylim([min(0, 0.8*min_point), max_point*1.2])
    plt.title('Lattice trapping frequency measurement')
    plt.savefig(result_path+'.svg', bbox_inches='tight')
    
    plt.clf()
    plt.close('all')
    # plt.show()
    
# def direct_plot():
#     """copy this script to the result folder."""
#     try:
#         current_folder = os.getcwd()
#         data_path = os.path.join(current_folder, 'processed_data.txt')
        
#         with open(data_path)
    
#     except:
#         pass








