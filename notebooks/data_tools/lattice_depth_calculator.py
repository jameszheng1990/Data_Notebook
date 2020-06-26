import numpy as np

m = 88*1.672e-27

beam_waist = 110e-6

h= 6.672e-34
hbar = h/(2*np.pi)
lambda_lat = 813.1e-9
k_lat = 2*np.pi/lambda_lat
erec = (hbar*k_lat)**2/(2*m)
c = 3e8

epsilon0 = 8.854e-12
a0 = 5.29e-11 #bohr radius
alpha = 290*4*np.pi*epsilon0*a0**3

def nuz_to_u(nuz):
    nuz = nuz*1e3
    u = (2*np.pi*nuz/k_lat)**2*m/2
    return round(u/erec, 2)

def uz(p1, p2, waist):
    u_z = 4*alpha*np.sqrt(p1*p2)/(np.pi*c*epsilon0 * waist**2)
    # u_const = alpha*(p1+p2-np.sqrt(p1*p2))/(np.pi*c*epsilon0*waist**2)
    return round(u_z/erec, 2)



nuz0 = 33
print('Trap depth is {} Erec for {} kHz nu_z.'.format(nuz_to_u(nuz0), nuz0))

p1 = 0.6
p2 = 0.4
print('Trap depth is {} Erec for {} W entrance and {} W retro power.'.format(uz(p1, p2, beam_waist), p1, p2))