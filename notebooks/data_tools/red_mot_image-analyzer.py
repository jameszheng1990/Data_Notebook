import os, h5py, re, time, json
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from camera.clients.data_tools.process_image import process_images_g
from camera.clients.data_tools.process_image import process_images_g_fluorescence
from matplotlib.offsetbox import AnchoredText

import sys
sys.path.insert(1, 'C:\\LabRad\\SrData\\data\\notebooks\\data_tools')
from helper import process_units
# from data.notebooks.helper.data_tools import process_camera

PROJECT_DATA_PATH = os.path.join(os.getenv('LABRADDATA'), 'data')

exp_name = 'red_mot_image'
    
def fit_Gaussian(x, x0, w, amp, const):
    return amp * np.exp(-pow(x - x0, 2)/(2*pow(w, 2))) + const

def process_red_mot_image_ccd(settings):
    shot = settings['shot']
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
            
        conductor_json_path = data_folder + '\\{}.conductor.json'.format(shot)
        ccd_hdf5_path = data_folder + '\\{}.{}.hdf5'.format(shot, method)
        
        with open(conductor_json_path, 'r') as file:
            f1 = file.read()
            f2 = json.loads(f1)
        try:
            em_gain = f2['andor.record_path']['em_gain']
        except:
            em_gain = 1
        
        try:
            images = {}
            with h5py.File(ccd_hdf5_path, "r") as images_h5:
                for key in images_h5:
                    images[key] = np.array(images_h5[key], dtype = 'float64')
                images_h5.close()
            
            if image == 'absorption':
                n0 = process_images_g(images, em_gain)
                x_trace0 = np.sum(n0, axis = 0)
                y_trace0 = np.sum(n0, axis = 1)
                xc = x_trace0.argmax()
                yc = y_trace0.argmax()
                n = n0[yc - roi_width: yc + roi_width, xc - roi_width: xc + roi_width]
                n = np.flipud(n)
                
                X = range(n.shape[1])
                Y = range(n.shape[0])
            
                # x_trace = np.sum(n, axis = 0)
                # y_trace = np.sum(n, axis = 1)
                            
                # p0x = (x_trace.argmax(), fit_width, np.max(x_trace), np.min(x_trace))
                # p0y = (y_trace.argmax(), fit_width, np.max(y_trace), np.min(y_trace))
                
                # try:
                #     poptx, pcovx =  optimize.curve_fit(fit_Gaussian, X, x_trace, p0 = p0x)
                #     popty, pcovy =  optimize.curve_fit(fit_Gaussian, Y, y_trace, p0 = p0y)
                    
                # except Exception as e1:
                #     print(e1)
                #     poptx = p0x
                #     popty = p0y
                        
                # count1 = np.sum(x_trace) - poptx[-1]*len(x_trace) # test
                # count2 = np.sum(y_trace) - popty[-1]*len(y_trace) # test
                # count = (count1+count2)/2
                # count = round(count, 1)
                count = np.sum(n)
                count = round(float(count), 1)
                
            elif image == 'fluorescence':
                n0 = process_images_g_fluorescence(images)
                (yc, xc) = np.unravel_index(np.argmax(n0, axis = None), n0.shape)
                n = n0[yc - roi_width: yc + roi_width, xc - roi_width: xc + roi_width]
                n = np.flipud(n)
            
                X = range(n.shape[1])
                Y = range(n.shape[0])
                count = float(np.sum(n))
                count = round(count, 1)
            
            data.append({x_key: shot, 'fit': {y_key: count}})
            
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(20, 8)
        
            cmap = plt.get_cmap('jet')
            ax.set_aspect('equal')
            plt.pcolormesh(X, Y, n, cmap = cmap)
            plt.colorbar()
            plt.title('Atom number: {:.2e}'.format(count))
                
            plt.savefig(save_path + '.png', bbox_inches='tight')
                
            plt.clf()
            plt.close('all')
        
        except Exception as e:
            print(e)
            count = 0
            data.append({x_key: shot, 'fit': {y_key: count}})
            
        return  data
    
    else:
        return data

def plot_red_mot_image_ccd(data, settings):

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
    x_data = [i[x_key] for i in sorted_data if min(data_range) < i['fit'][y_key] < max(data_range) ]
    y_data = [i['fit'][y_key] for i in sorted_data if min(data_range) < i['fit'][y_key] < max(data_range)]
    
    mean = np.mean(y_data)
    mean = round(float(mean), 1)
    std = np.std(y_data)
    std = round(float(std), 1)
    
    with open(saved_data_path, 'w') as file:
        file.write(json.dumps(sorted_data))
    
    x_data = process_units(x_data, units)
    
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(20, 8)
        
    ax.plot(x_data, y_data)
    ax.plot(x_data, y_data, 'ro')
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label + ' [{}]'.format(units))
    anchored_text = AnchoredText('mean = {}, std = {}'.format(mean, std) , loc=2)
    ax.add_artist(anchored_text)
    # max_point = max([max(l.get_ydata()) for l in ax.get_lines()])
    # min_point = min([min(l.get_ydata()) for l in ax.get_lines()])
    
    # ax[0].set_ylim([min(0, 0.8*min_point), max_point*1.2])
    plt.title('Red mot image')
    plt.savefig(result_path+'.png', bbox_inches='tight')
    
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








