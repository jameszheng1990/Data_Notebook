import os, h5py, re, time, json
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from camera.clients.data_tools.process_image import process_images_absorption_images
# from data.notebooks.helper.data_tools import process_camera

PROJECT_DATA_PATH = os.path.join(os.getenv('LABRADDATA'), 'data')

exp_name = 'lattice_trap_freq_modulation'

units_table = {'Hz': 1, 'kHz': 1e3, 'MHz': 1e6, 'GHz': 1e9, 'THz': 1e12}

def process_freq(x_data, units):
    try:
        factor = units_table[units]
        return [i/factor for i in x_data]
    except:
        return x_data
    
def fit_Gaussian(x, x0, w, amp, const):
    return amp * np.exp(-pow(x - x0, 2)/(2*pow(w, 2))) + const

def process_lattice_trap_freq_modulation_tcam(settings):
    shot = settings['shot']
    roi_width = settings['kwargs']['roi_width']
    fit_width = settings['kwargs']['fit_width']
    
    if shot >= 0:
        data = []
        data_folder = os.path.join(PROJECT_DATA_PATH, settings['data_path'])
        save_folder = os.path.join(PROJECT_DATA_PATH, settings['data_path'], 'processed_data')
        save_path = os.path.join(save_folder, str(shot))
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
            
        conductor_json_path = data_folder + '\\{}'.format(shot) + '.conductor.json'
        tcam_hdf5_path = data_folder + '\\{}'.format(shot) + '.tcam.hdf5'
        
        try:
            f = open(conductor_json_path, 'r')
            f1 = f.read()
            f2 = json.loads(f1)
            freq = f2['lattice.ModulationFrequency']
            
            images = {}
            with h5py.File(tcam_hdf5_path, 'r') as images_h5:
                for key in images_h5:
                    images[key] = np.array(images_h5[key], dtype = 'float64')
                images_h5.close()
        
        
            n0 = process_images_absorption_images(images)
            yc, xc = np.unravel_index(n0.argmax(), n0.shape)
            n = n0[yc - roi_width: yc + roi_width, xc - roi_width: xc + roi_width]
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
            
            count1 = np.sum(x_trace) - poptx[-1]*len(x_trace) # test
            count2 = np.sum(y_trace) - popty[-1]*len(y_trace) # test
            count = (count1+count2)/2
            
            data.append({'shot': shot, 'modulation_frequency':freq, 'tcam_count': count})
            
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(20, 8)
        
            cmap = plt.get_cmap('jet')
            ax.set_aspect('equal')
            plt.pcolormesh(X, Y, n, cmap = cmap, vmin = 0)
            plt.colorbar()
            plt.title('Atom number: {:.2e}'.format(count))
                
            plt.savefig(save_path + '.svg', bbox_inches='tight')
                
            plt.clf()
            plt.close('all')
        
        except:
            count = 0
            data.append({'shot': shot, 'modulation_frequency':freq, 'tcam_count': count})
            
        return  data
    
    else:
        return data

def plot_lattice_trap_freq_modulation_tcam(data, settings):
    # print(data)

    units = settings['kwargs']['units']
    data_range = settings['kwargs']['data_range']
    
    result_folder = os.path.join(PROJECT_DATA_PATH, settings['data_path'], 'result')
    result_path = os.path.join(result_folder, os.path.split(settings['data_path'])[0] + '_' + os.path.split(settings['data_path'])[1])
    saved_data_path = os.path.join(result_folder, 'processed_data.txt')
    
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)
    
    # sort by shot number
    sorted_data = sorted(data, key = lambda k:k['shot'])
    x_data = [i['modulation_frequency'] for i in sorted_data if min(data_range) <= i['tcam_count'] < max(data_range) ]
    y_data = [i['tcam_count'] for i in sorted_data if min(data_range) <= i['tcam_count'] < max(data_range)]
    
    with open(saved_data_path, 'w') as file:
        file.write(json.dumps(sorted_data))
    
    x_data = process_freq(x_data, units)
    
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(20, 8)
        
    ax.plot(x_data, y_data)
    ax.set_ylabel('atom number (a.u.)')
    ax.set_xlabel('modulation frequency ({})'.format(units))
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








