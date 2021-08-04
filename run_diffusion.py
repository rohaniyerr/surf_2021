#!/usr/bin/env python
# coding: utf-8

# ## **Diffusion Equation**

# $$
# \frac{d\Sigma}{dt} = \frac{3}{r}\frac{\partial}{\partial r} \left[r^{1/2} \frac{\partial}{\partial r}(\nu \Sigma r^{1/2})\right]
# $$
# 
# This is evolution equation for the surface density of a geometrically thin disk
# 
# Putting this in the form of a diffusion equation:
# 
# $$
# \frac{\partial f}{\partial t} = D \frac{\partial^2 f}{\partial X^2}
# $$
# where $X \equiv 2r^{1/2}$, $f \equiv \frac{3}{2}\Sigma X$ and $D = \frac{12\nu}{X^2}$
# 

# We need numerical values for $\nu$
# $$
# \nu = \alpha c_s H \\
# H = \frac{c_s}{\Omega} \\
# \Omega = \sqrt{\frac{G*M_s}{r^3}} \\
# c_s = \sqrt{\frac{k_B * T}{\mu * mH}} \\
# mH = \frac{1}{N_A} \\
# \mu = 2.3 \\
# $$
# 
# The Temperature profile is given by:
# $$
# 2 \sigma_b T_{disk}^4 = \frac{9}{4}\Sigma \nu \Omega^2\\
# T^3 = \frac{9}{8\sigma_b}\Sigma \frac{k_b \alpha}{\mu mH}\sqrt{\frac{GM}{r^3}}
# $$
# 
# The Pressure is given by:
# 
# $$
# P = \frac{c_s^2\Sigma}{H} = c_s\Sigma\Omega
# $$
# 

# plot feature
import numpy as np
import numba
import os
import time

from dust_diffuse import *
from thermal import *


# In[2]:


# Natural Constants
mu   = 2.2         # mean molar mass in disk
avog = 6.022142e3  # avogadros number
mH = 1.673534e-27  # atomic hydrogen mass
kB = 1.380649e-23  # boltzmann constant
G  = 6.6738e-11    # gravitational constant
Ms = 1.9886e30     # solar mass
Me = 5.972e24      # earth mass
AU = 1.4959787e11  # astronomical units
yrs2sec = 3.1536e7 # convert year to seconds
sb = 5.6704e-8

# Disk Parameters
rin  = 0.05 * AU       # inner radius of the disk
rout = 30 * AU         # outer radius of the disk
sigma_in  = 2e5        # initial sigma_gas at 1AU
sigma_max = sigma_in*2 # initial sigma_gas at r > rcrit
sigma_min = 1e2
distance = rout - rin  # distance from inner radius to outer radius

# Temporal discretization
dyr = .1
dt  = dyr * yrs2sec # timestep

max_years  = 2e5
final_time = max_years*yrs2sec + 1 #total diffusion time in seconds
t          = np.arange(0, final_time, dt)

t_save_interval = 1000


# Spacial discretization
n  = 600        # number of steps / number of points to plot
dr = distance/n # distance between each spacial step


# Pebble Parameters
Q     = 1.
St    = .01 #Stokes Number
Stp   = .05
tau_g = 1e5 # in years

cor_ratio  = .1
calc_ratio = .1
for_ratio  = .8

cont = False




#Initialize grid spaces

dist  = np.empty(n)
Omega = np.empty(n)
X     = np.empty(n)


#Initialize Staggered Grid

dist_stag  = np.empty(n-1)
Omega_stag = np.empty(n-1)
dPdr       = np.empty(n-1)
cs2_stag   = np.empty(n-1)
press_stag = np.empty(n-1)


#Boundary Conditions

f_in = 0
f_out = 0




@numba.njit
def make_alpha_grid(cont):
    if(cont):
        alphas = np.linspace(1e-2, 1e-3, n)
    else:
        alphas = np.ones(n) * 1e-3
    return alphas


def update_alpha(alphas, T):
    alphas = [alpha_MRI_hydro(temp) for temp in T]
    return np.array(alphas)

def calc_dist_omega_stag():
    for i in range(1, n):
        dist_stag[i-1] = 0.5 * (dist[i] + dist[i-1])
        Omega_stag[i-1] = np.sqrt(G * Ms / dist_stag[i-1] ** 3)
        

def update_stag_grid(alphas, cs2):
    cs2_stag = np.array([0.5 * (cs2[i] + cs2[i-1]) for i in range(1, n)])
    alphas_stag = np.array([0.5 * (alphas[i] + alphas[i-1]) for i in range(1, n)])
    return (alphas_stag, cs2_stag)

@numba.njit
def update_nu(cs2, alphas):
    return alphas*cs2/Omega

@numba.njit
def update_St(cs2, cs2_stag, alphas, alpha_stag):
    dust = Q/(cs2_stag*alpha_stag)
    pebble = Q*10/(cs2*alphas)
    return (dust, pebble)

# In[14]:


def calc_init_params():
    """
    Calculate the initial parameters values for time t = 0.
    """
    sigma_gas    = np.empty(n)
    sigma_dust   = np.empty((3,n))
    sigma_pebble = np.empty((3,n))
    sigma_evap_init = np.zeros((3,n))
    alphas = make_alpha_grid(cont)
    for i in range(n):
        dist[i]  = (rin + (rout-rin)*i/(n-1))
        X[i]     = 2 * np.sqrt(dist[i])
        Omega[i] = np.sqrt(G * Ms / (dist[i] ** 3))

        # gas/dust density
        sigma_gas[i] = sigma_in * AU / dist[i]
        if (sigma_gas[i]>sigma_max):
            sigma_gas[i] = sigma_max
        if (dist[i]/AU > 15):
            sigma_gas[i] = sigma_min
        sigma_dust[0,i] = sigma_gas[i] * 0.005 * cor_ratio
        sigma_dust[1,i] = sigma_gas[i] * 0.005 * calc_ratio
        sigma_dust[2,i] = sigma_gas[i] * 0.005 * for_ratio

    cs2, T, P, sigma_dust, sigma_evap, sigma_pebble = calc_thermal_struc(sigma_gas, sigma_dust, sigma_pebble, sigma_evap_init, alphas, Omega)
    calc_dist_omega_stag()
    alphas = update_alpha(alphas, T)
    alphas_stag, cs2_stag = update_stag_grid(alphas, cs2)
    St       = Q/(alphas_stag * cs2_stag)
    nu       = alphas*cs2/Omega
    v_gas    = calc_gas_vel(nu, sigma_gas, dist, dist_stag)
    v_dust   = calc_dust_vel(St, cs2_stag, v_gas, P)
    v_pebble = v_dust*10 #calc_dust_vel(Stp,cs2_stag, v_gas, P)
    D = 12 * nu / (X ** 2)
    f = (1.5 * X * sigma_gas)
    return (sigma_gas, sigma_dust, sigma_evap, sigma_pebble, v_gas, v_dust, v_pebble, cs2, nu, D, f, alphas)

@numba.njit
def calc_dust_vel(St, cs2_stag, v_gas, press):
    press_stag = np.array([0.5*(press[i]+press[i-1]) for i in range(1,n)])
    dPdr = np.array([(press[i] - press[i-1])/dist_stag[i-1] for i in range(1,n)])
    rho_stag = press_stag/cs2_stag
    v_dust = (St/(1+(St)**2)) * (1 / (rho_stag * Omega_stag))*dPdr
    v_dust = np.append(v_dust, 0)
    
#     v_dust = v_dust + v_gas
    return v_dust

@numba.njit
def calc_pebble_growth(sigma_pebble, sigma_dust):
    tau = 1.988e-7 * tau_g *yrs2sec / Omega
    sigma_pebble += sigma_dust * dt / tau
    sigma_dust -= sigma_dust * dt / tau
    return (sigma_pebble, sigma_dust)

@numba.njit
def calc_gas_evol(f, dt, D):
    """Outputs the surface density at a specific dt."""
    df_dt = np.empty(n)
    for j in range(1, n-1):
        dX1 = X[j] - X[j-1]
        dX2 = X[j+1] - X[j]
        D1 = 0.5 * (D[j] + D[j-1])
        D2 = 0.5 * (D[j+1] + D[j])
        df_dt[j] = D1 * ((-(f[j] - f[j-1])/dX1**2)) + D2 * ((f[j+1]-f[j])/dX2**2)
    dX_final = X[-1]-X[-2]
    dX_in = X[1]-X[0]
    df_dt[0] = D[0] * (-(f[0] - f_in)/dX_in**2 + (f[1]-f[0])/dX_in**2)
    df_dt[n-1] = D[n-1] * (-(f[n-1] - f[n-2])/dX_final**2 + (f_out-f[n-1])/dX_final**2)
    f_new = f + df_dt * dt
    sigma_at_time = [2*f_new[k]/(3*X[k]) for k in range(n)]
    return (f_new, sigma_at_time)


# In[17]:


# @numba.njit
def calc_disk_mass(sigma):
    """Calculate the mass at a specific dt"""
    disk_mass = 0
    for j in range(n):
        disk_mass += 2 * np.pi * dist[j] * dr * sigma_gas[j]
        disk_mass += 2 * np.pi * dist[j] * dr * sigma_dust[j]
#     print(f'The mass of the disk is {disk_mass}.')
    return disk_mass


# In[18]:


def save2dir(**alpha_run):
    output_dir = 'output-aMRI-tau1e4-'+str(tau_g)+'/'
    filename = output_dir + 'disk_'
    return (output_dir, filename)


def update_nu(cs, alphas):
    return alphas*cs2/Omega

@numba.njit
def update_D(nu):
    return 12 * nu / (X ** 2)

#Time Evolution Main

sigma_gas, sigma_dust, sigma_evap, sigma_pebble, v_gas, v_dust, v_pebble, cs2, nu, D, f, alphas = calc_init_params()
output_dir, filename = save2dir()

# sigma_evap = np.zeros(n)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
start = time.time()
for i in range(len(t)):
    f_new, sigma_gas = calc_gas_evol(f, dt, D)
    sigma_pebble, sigma_dust = calc_pebble_growth(sigma_pebble, sigma_dust)
    for el in range(3):
        sigma_dust[el,:]   = calc_evol(sigma_gas, sigma_dust[el,:], nu, v_dust, dist, dt)
        sigma_pebble[el,:] = calc_evol(sigma_gas, sigma_pebble[el,:], nu, v_pebble, dist, dt)
        sigma_evap[el,:]   = calc_evol(sigma_gas, sigma_evap[el,:], nu, v_gas, dist, dt)
    cs2, T, P, sigma_dust, sigma_evap, sigma_pebble = calc_thermal_struc(sigma_gas, sigma_dust, sigma_pebble, sigma_evap, alphas, Omega)
    alphas   = update_alpha(alphas, T)
    nu       = update_nu(cs2, alphas)
    D        = update_D(nu)
    alphas_stag, cs2_stag = update_stag_grid(alphas, cs2)
    St_dust, St_pebble    = update_St(cs2, cs2_stag, alphas, alphas_stag)
    v_dust   = calc_dust_vel(St_dust, cs2_stag, v_gas, P)
    v_pebble = v_dust*10 #calc_dust_vel(Stp,     cs2_stag, v_gas, P) #alc_pebble_vel(v_dust)
    v_gas    = calc_gas_vel(nu, sigma_gas, dist, dist_stag)

    f = f_new
    
    if (i%(t_save_interval/dyr) == 0):
        print(i*dyr)
        output = np.vstack([dist/AU, sigma_gas, sigma_dust, sigma_evap, sigma_pebble, v_dust, T, P])
        output = np.transpose(output)
        np.savetxt(filename + str(int(i*dyr)) + '.txt', output, delimiter=',', newline='\n')# if you want to save results to file

        end   = time.time()
        print(end-start)
        start = time.time()




