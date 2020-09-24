import os, h5py, re, time
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

from camera.clients.data_tools.process_image import process_images_g_fluorescence

file_path = os.path.join(os.getcwd(), "0.ixon.hdf5")

images = {}
with h5py.File(file_path, 'r') as images_h5:
    for key in images_h5:
        images[key] = np.array(images_h5[key], dtype = 'float64')
    images_h5.close()
                
n0 = process_images_g_fluorescence(images)
x_trace0 = np.sum(n0, axis = 0)
y_trace0 = np.sum(n0, axis = 1)
xc = x_trace0.argmax()
yc = y_trace0.argmax()
xwid = 20
ywid = 20
n = n0[yc - ywid: yc + ywid, xc - xwid: xc + xwid]
n = np.flipud(n)

def fit_func(x, x0, w, a, c):
    return a*np.exp(-pow(x - x0, 2)/(2*pow(w, 2))) + c

X = range(n.shape[1])
Y = range(n.shape[0])

x_trace = np.sum(n, axis = 0)
y_trace = np.sum(n, axis = 1)

p0x = (x_trace.argmax(), 5, np.max(x_trace), np.min(x_trace))
p0y = (y_trace.argmax(), 5, np.max(y_trace), np.min(y_trace))
            
poptx, pcovx =  optimize.curve_fit(fit_func, X, x_trace, p0 = p0x)
popty, pcovy =  optimize.curve_fit(fit_func, Y, y_trace, p0 = p0y)

xcen = poptx[0]
ycen = popty[0]
xwidth = poptx[1]
ywidth = popty[1]

plt.figure()
t_plot = np.linspace(0, np.max(X), 100)
plt.scatter(X, x_trace, c = 'blue')
plt.plot(t_plot, fit_func(t_plot, poptx[0], poptx[1], poptx[2], poptx[3]))
plt.figure()
t_plot = np.linspace(0, np.max(Y), 100)
plt.scatter(Y, y_trace, c = 'red')
plt.plot(t_plot, fit_func(t_plot, popty[0], popty[1], popty[2], popty[3]))

n_roi = n[ int(ycen - 5*ywidth): int(ycen + 5*ywidth), int(xcen - 5*xwidth): int(xcen + 5*xwidth)] 
X_roi = range(n_roi.shape[1])
Y_roi = range(n_roi.shape[0])

plt.figure()
ax = plt.subplot()
cmap = plt.get_cmap('jet')
ax.set_aspect('equal')

X = range(n0.shape[1])
Y = range(n0.shape[0])
x_trace = np.sum(n0, axis = 0)
y_trace = np.sum(n0, axis = 1)
plt.pcolormesh(X, Y, n0, cmap = cmap, vmin = 0)

plt.plot(X, x_trace/max(x_trace)*xwid, c = 'yellow')
plt.plot(y_trace/max(y_trace)*ywid, Y , c = 'yellow')
# plt.pcolormesh(X_roi, Y_roi, n_roi, cmap = cmap)
plt.colorbar()


print('Total atom number is {:.2e}'.format(np.sum(x_trace) - len(x_trace)*poptx[-1] ))