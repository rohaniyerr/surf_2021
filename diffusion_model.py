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

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import numba
import pandas as pd
import os
from dust_diffuse import *
from thermal import *


# In[2]:


#Natural Constants

# T = 600 # Temperature
mu = 2.2 # mean molar mass in disk
avog = 6.02214 * 10**23 # avogadros number
mH = 1.673534 * 10**-27 # atomic hydrogen mass
kB = 1.380649 * 10**-23 #boltzmann constant
G = 6.6738 * 10**-11 #gravitational constant
Ms = 1.9886 * 10**30 #solar mass
Me = 5.972 * 10 **24 #earth mass
AU = 1.4959787 * 10**11 #astronomical units
yrs2sec = 3.1536 * 10**7 #convert years to seconds
sb = 5.6704*10**-8


# In[3]:


#Disk Parameters
alpha1 = 1e-2
alpha2 = 1.0 * 10**-3
alpha3 = 1.0 * 10**-4
rin = 0.05 * AU #inner radius of the disk
rout = 30 * AU #outer radius of the disk
sigma_in  = 2 * 10**5 #boundary condition at t=0
sigma_max = sigma_in*2 #boundary condition at t=final_time
sigma_min = 1 * 10**2
distance = rout - rin #distance from inner radius to outer radius


# In[4]:


#Temporal discretization
max_years = 10
dyr = .1
dt = dyr * yrs2sec #timestep
final_time = max_years*yrs2sec + 1 #total diffusion time in seconds
t_save_interval = 1 #every ten years
t_plot = [0,8]


# In[5]:


#Spacial discretization

n = 300 #number of steps / number of points to plot
dr = distance/n #distance between each spacial step
cont = True #Boolean for continuous alpha distribution


# In[6]:


#Other Parameters

St = .01 #Stokes Number
Tcor = 1800 #10% corundum
Tcalc = 1700 #10% calcium
Tfor = 1400 #80% forsterite

cor_ratio = .1
calc_ratio = .1
for_ratio = .8

phase_change = 1400


# In[7]:


#Plotting axes
t = np.arange(0, final_time, dt)


# In[8]:


#Initialize grid spaces

dist = np.empty(n)
Omega = np.empty(n)
X = np.empty(n)


# In[9]:


#Initialize Staggered Grid

dist_stag = np.empty(n-1)
Omega_stag = np.empty(n-1)
dPdr = np.empty(n-1)
cs2_stag = np.empty(n-1)
press_stag = np.empty(n-1)
rho_stag = np.empty(n-1)


# In[ ]:





# In[10]:


#Boundary Conditions

f_in = 0
f_out = 0


# In[11]:


@numba.njit
def make_alpha_grid(cont):
    if(cont):
        alphas = np.linspace(alpha1, alpha3, n)
        return alphas
    else:
        alphas = np.empty(n)
        for i in range(n):
            if i < (n//3):
                alphas[i] = alpha1
            elif i < (2*n//3):
                alphas[i] = alpha2
            else:
                alphas[i] = alpha3


# In[12]:


def update_alpha(alphas, T):
    for idx, temp in enumerate(T):
        if(temp >= phase_change):
            alphas[idx] = alpha2 + 9e-3*min((temp-phase_change)/100, 1.)
        else:
            alphas[idx] = alpha2
    return alphas


# In[13]:


def calc_stag_grid(T):
    for i in range(1, n):
        dist_stag[i-1] = 0.5 * (dist[i] + dist[i-1])
        Omega_stag[i-1] = np.sqrt(G * Ms / dist_stag[i-1] ** 3)
        Tmd = 0.5 * (T[i] + T[i-1])
        cs2_stag[i-1] = (kB * Tmd)/(mu*mH)


# In[14]:


def calc_init_params():
    """
    Calculate the initial parameters values for time t = 0.
    """
    sigma_gas = np.empty(n)
    sigma_dust = np.empty(n)
    sigma_evap_init = np.zeros(n)
    alphas = make_alpha_grid(cont)
    for i in range(n):
        dist[i] = (rin + (rout-rin)*i/(n-1))
        sigma_gas[i] = sigma_in * AU / dist[i]
        if (sigma_gas[i]>sigma_max):
            sigma_gas[i] = sigma_max
        if (dist[i]/AU > 15):
            sigma_gas[i] = sigma_min
        sigma_dust[i] = sigma_gas[i] * 0.005
        X[i] = 2 * np.sqrt(dist[i])
        Omega[i] = np.sqrt(G * Ms / (dist[i] ** 3))
    cs2, T, P = calc_thermal_struc(sigma_gas, sigma_dust, sigma_evap_init, alphas, Omega)
    calc_stag_grid(T)
    alphas = update_alpha(alphas, T)
    nu = alphas*cs2/Omega
    v_gas = calc_gas_vel(nu, sigma_gas, dist, dist_stag)
    v_dust = calc_dust_vel(v_gas, P)
    D = 12 * nu / (X ** 2)
    f = (1.5 * X * sigma_gas)
    return (sigma_dust, sigma_gas, v_dust, v_gas, cs2, nu, D, f, alphas)


# In[ ]:





# In[15]:


def calc_dust_vel(v_gas, press):
    press_stag = [0.5*(press[i]+press[i-1]) for i in range(1,n)]
    dPdr = [(press[i] - press[i-1])/dist_stag[i-1] for i in range(1,n)]
    rho_stag = press_stag/cs2_stag
    v_dust = (St/(1+(St)**2)) * (1 / (rho_stag * Omega_stag))*dPdr
    v_dust = np.append(v_dust, 0)
#     v_dust = v_dust + v_gas
    return v_dust


# In[16]:


@numba.njit
def calc_gas_evol(f, dt):
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
    if(alpha_run):
        if(alpha_run == alpha1):
            output_dir = 'output_a1/'
            filename = 'output_a1/disk_'
        elif(alpha_run == alpha2):
            output_dir = 'output_a2/'
            filename = 'output_a2/disk_'
        else:
            output_dir = 'output_a3/'
            filename = 'output_a3/disk_'
    else:
        output_dir = 'output/'
        filename = 'output/disk_'
    return (output_dir, filename)


# In[19]:



# In[21]:


def update_nu(cs):
    return alphas*cs2/Omega


# In[22]:


#Time Evolution Main

sigma_dust, sigma_gas, v_dust, v_gas, cs2, nu, D, f, alphas = calc_init_params()
output_dir, filename = save2dir()

sigma_evap = np.zeros(n)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for i in range(len(t)):
    f_new, sigma_gas = calc_gas_evol(f, dt)
    sigma_dust = calc_evol(sigma_gas, sigma_dust, nu, v_dust, dist, dt)
    sigma_evap = calc_evol(sigma_gas, sigma_evap, nu, v_gas, dist, dt)
    cs2, T, P = calc_thermal_struc(sigma_gas, sigma_dust, sigma_evap, alphas, Omega)

    for j in range(len(sigma_gas)):
        (tmp1, tmp2) = evap_cond(sigma_dust[j], sigma_evap[j], T[j])
        sigma_dust[j] = tmp1
        sigma_evap[j] = tmp2
        
    alphas = update_alpha(alphas, T)
#     sigma_evap = evaporate(sigma_dust, T)
#     sigma_dust = condense(sigma_evap, sigma_dust, T)
    nu = update_nu(cs2)
    calc_stag_grid(T)
    v_dust = calc_dust_vel(v_gas, P)
    v_gas = calc_gas_vel(nu, sigma_gas, dist, dist_stag)
    f = f_new
    if (i%(t_save_interval/dyr) == 0):
        output = np.vstack([dist/AU, sigma_gas, sigma_dust, sigma_evap, v_dust, T, P])
        output = np.transpose(output)
        np.savetxt(filename + str(int(i*dyr)) + '.txt', output, delimiter=',', newline='\n')# if you want to save results to file


# In[23]:


def plot():
    #get_ipython().run_line_magic('matplotlib', '')
    for time in t_plot:
        print(time)
        df = pd.read_csv('output/disk_' + str(int(time)) + '.txt', delimiter=',', header=None)
        sigma_gas_plot = df[1].to_numpy().astype(np.float)
        sigma_dust_plot = df[2].to_numpy().astype(np.float)
        temp_plot = df[5].to_numpy().astype(np.float)
        dg_ratio = sigma_dust_plot/sigma_gas_plot
#         plt.figure(1)
#         plt.title(f'Surface Density Evolution at t={int(max_years)} years')
#         plt.xlabel('Distance (AU)')
#         plt.ylabel('Surface Density')
#         plt.yscale('log')
#         plt.plot(dist/AU, sigma_dust_plot)
#         plt.plot(dist/AU, sigma_gas_plot)
        plt.figure(2)
        plt.title(f'Temperature Evolution at t={int(max_years)} years')
        plt.xlabel('Distance (AU)')
        plt.ylabel('Temperature')
        plt.yscale('log')
        plt.plot(dist/AU, temp_plot)
        plt.figure(3)
        plt.title(f'Dust to Gas Ratio Evolution at t={int(max_years)} years')
        plt.xlabel('Distance (AU)')
        plt.ylabel('Dust/Gas')
        plt.yscale('log')
        plt.plot(dist/AU, dg_ratio)
    plt.show()


# In[24]:


plot()


# In[ ]:





# In[ ]:





# In[ ]:




