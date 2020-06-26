import os, h5py, re, time, json
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from camera.clients.data_tools.process_image import process_images_absorption_images
# from data.notebooks.helper.data_tools import process_camera

PROJECT_DATA_PATH = os.path.join(os.getenv('LABRADDATA'), 'data')

exp_name = 'lattice_lifetime'

time_units_table = {'s': 1, 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9}

def process_time(x_data, units):
    try:
        factor = time_units_table[units]
        return [i/factor for i in x_data]
    except:
        return x_data
    
def fit_Gaussian(x, x0, w, amp, const):
    return amp * np.exp(-pow(x - x0, 2)/(2*pow(w, 2))) + const

def fit_exp_decay(t, a, tau):
    return a* np.exp(-t/tau)

def process_lattice_lifetime_tcam(settings):
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
            holdtime = f2['sequencer.DO_parameters.lattice_hold']
            
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
            
            data.append({'shot': shot, 'lattice_hold':holdtime, 'tcam_count': count})
            
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
        
        except Exception as e2:
            print(e2)
            count = 0
        
        data.append({'shot': shot, 'lattice_hold':holdtime, 'tcam_count': count})
        return  data

    else:
        return data

def plot_lattice_lifetime_tcam(data, settings):
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
    x_data = [i['lattice_hold'] for i in sorted_data if min(data_range) <i['tcam_count'] < max(data_range) ]
    y_data = [i['tcam_count'] for i in sorted_data if min(data_range) <i['tcam_count'] < max(data_range)]
    
    with open(saved_data_path, 'w') as file:
        file.write(json.dumps(sorted_data))
    
    x_data = process_time(x_data, units)
    
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

    t1 = np.arange(0, np.max(x_data)+ 5, 0.1)

    plt.figure()
    plt.ylabel("Atom number (arb. units)")
    plt.xlabel("Lattice hold time ()".format(units))
    plt.title('Lattice lifetime measurement')
    plt.scatter(x_data, y_data, c='red', marker = 's', s= 35)
    plt.plot(t1, fit_exp_decay(t1, popt[0], popt[1]), label= str(round(popt[1], 2 ))+'s' , c='red',
             linestyle='dashed')
    plt.legend(loc='best')
    
    
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







# ###############################


# import os, h5py, re, time, json
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import optimize
# from camera.clients.data_tools.process_image import process_images_absorption_images

# exp_name = 'lattice_lifetime_measurement_hold'

# current_folder = os.getcwd()

# folder_list = os.listdir()
# folder_list = [i for i in folder_list if '#' in i and '.py' not in i]

# t_list = []
# n_list = []
# nroi_list = []
# X_roi_list = []
# Y_roi_list = []

# number_list = []

# x_trace_list = []
# y_trace_list = []


# pixel_size = 10.59 # [um]
# cross_section = 0.1014 # [um^2]
# linewidth = 201.06192983 # [ns?]
# pulse_length = 10 # [us]
# efficiency = 0.50348 
# gain = 1
# high_intensity_coefficient = 2 / (linewidth * pulse_length * efficiency * gain)
# low_intensity_coefficient = pixel_size**2 / cross_section

# t_delay = 0.1 # in ms

# def fit_Gaussian(x, x0, w, a, c):
#     return a*np.exp(-pow(x - x0, 2)/(2*pow(w, 2))) + c

# data_folder = current_folder
# roi_width = 30
# fit_width = 10

# xdata = []
# ydata = []

# for shot in range(5):
#     conductor_json_path = data_folder + '\\{}'.format(shot) + '.conductor.json'
#     tcam_hdf5_path = data_folder + '\\{}'.format(shot) + '.tcam.hdf5'
    
#     f = open(conductor_json_path, 'r')
#     f1 = f.read()
#     f2 = json.loads(f1)
#     t = f2['sequencer.DO_parameters.lattice_hold']
    
#     images = {}
#     with h5py.File(tcam_hdf5_path, 'r') as images_h5:
#         for key in images_h5:
#             images[key] = np.array(images_h5[key], dtype = 'float64')
#         images_h5.close()
            
#     n0 = process_images_absorption_images(images)
#     yc, xc = np.unravel_index(n0.argmax(), n0.shape)
#     n = n0[yc - roi_width: yc + roi_width, xc - roi_width: xc + roi_width]
#     n = np.flipud(n)
            
#     X = range(n.shape[1])
#     Y = range(n.shape[0])
                
#     x_trace = np.sum(n, axis = 0)
#     y_trace = np.sum(n, axis = 1)
                        
#     p0x = (x_trace.argmax(), fit_width, np.max(x_trace), np.min(x_trace))
#     p0y = (y_trace.argmax(), fit_width, np.max(y_trace), np.min(y_trace))
            
#     try:
#         poptx, pcovx =  optimize.curve_fit(fit_Gaussian, X, x_trace, p0 = p0x)
#         popty, pcovy =  optimize.curve_fit(fit_Gaussian, Y, y_trace, p0 = p0y)
            
#     except:
#         poptx = p0x
#         popty = p0y
            
#     count = np.sum(y_trace) - popty[-1]*len(y_trace) # test
            
#     xdata.append(t)
#     ydata.append(count)


# # Fitting lifetime

# def fit_func(t, a, tau):
#     return a* np.exp(-t/tau)

# x_list = xdata
# y_list = ydata

# x_data = np.asarray(x_list)
# y_data = np.asarray(y_list)

# y_data = y_data/(np.max(y_data))


# p0 = [max(y_list), 0.7*max(x_list)]

# popt, pcov = optimize.curve_fit(fit_func, x_data, y_data, p0 = p0)
# perr = np.sqrt(np.diag(pcov))

# print(popt, perr)

# lifetime = round(popt[1], 1)
# errorbar = round(perr[1], 1)


# t1 = np.arange(0, max(x_list)+ 5, 0.1)

# plt.figure()

# plt.ylabel("Atom number (arb. units)")
# plt.xlabel("Hold time (s)")
# plt.title('Lattice lifetime measurement')
# plt.scatter(x_data, y_data, c='red', marker = 's', s= 35)
# plt.plot(t1, fit_func(t1, popt[0], popt[1]), label= str(round( popt[1], 2 ))+'s' , c='red',
#           linestyle='dashed')

# plt.legend(loc='best')
# plt.show()

# plt.savefig('lattice lifetime measurement.svg', bbox_inches='tight')
