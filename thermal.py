import numpy as np
import numba


mH = 1.673534e-27
mu = 2.2 # mean molar mass in disk
kB = 1.380649e-23 #boltzmann constant
sb = 5.6704e-8
G = 6.6738 * 10**-11 #gravitational constant
Ms = 1.9886 * 10**30 #solar mass

cor_idx = 0
calc_idx = 1
for_idx = 2

cor_ratio = .1
calc_ratio = .1
for_ratio = .8

@numba.njit
def alpha_MRI_hydro(T):
    phase_change = 1400
    if(T >= phase_change):
        alpha = 1e-3 + 9e-3*min((T-phase_change)/100, 1.)
    else:
        alpha = 1e-3

    return alpha

@numba.njit
def evap_cond_dust(sigma_dust, sigma_evap, temp):
    Tcor = 1800
    Tcalc_start = 1700
    Tcalc_stop  = 1500
    Tfor        = 1400
    minit       = sigma_dust + sigma_evap
    cor_idx  = 0
    calc_idx = 1
    for_idx  = 2
    if (temp > Tcor): #temp > 1800, everything evaporates
        sigma_evap[:] = sigma_evap[:] + sigma_dust[:]
        sigma_dust[:] = 0
    elif (temp > Tcalc_start): #1700 < temp < 1800, calcium and forsterite evaporate, corundum condenses
        sigma_evap[cor_idx] =  minit[cor_idx] * (temp - Tcalc_start)/(Tcor - Tcalc_start)
        sigma_dust[cor_idx] =  minit[cor_idx] * (Tcor - temp)/(Tcor - Tcalc_start)
        sigma_evap[calc_idx] = minit[calc_idx]
        sigma_dust[calc_idx] = 0
        sigma_evap[for_idx] = minit[for_idx]
        sigma_dust[for_idx] = 0
    elif(temp > Tcalc_stop):  # 1500 < temp < 1700, forsterite evaporates, corundum condense
        sigma_evap[calc_idx] =  minit[calc_idx] * (temp - Tcalc_stop)/(Tcalc_start - Tcalc_stop)
        sigma_dust[calc_idx] = minit[calc_idx] * (Tcalc_start - temp)/(Tcalc_start - Tcalc_stop)
        sigma_dust[cor_idx] = minit[cor_idx]
        sigma_evap[cor_idx] = 0
        sigma_evap[for_idx] = minit[for_idx]
        sigma_dust[for_idx] = 0
    elif(temp > Tfor):  # 1400 < temp < 1500, forstertite evaporates, calcium and corundum condense
        sigma_evap[for_idx] =  minit[for_idx] * (temp - Tfor)/(Tcalc_stop - Tfor)
        sigma_dust[for_idx] = minit[for_idx] * (Tcalc_stop - temp)/(Tcalc_stop - Tfor)
        sigma_dust[cor_idx] = minit[cor_idx]
        sigma_evap[cor_idx] = 0
        sigma_dust[calc_idx] = minit[calc_idx]
        sigma_evap[calc_idx] = 0
    else: #temp < 1400, everything condenses
        sigma_dust[:] = sigma_dust[:] + sigma_evap[:]
        sigma_evap[:] = 0
    return (sigma_dust, sigma_evap)

@numba.njit
def evap_cond_pebble(sigma_pebble, sigma_evap, temp):
    Tcor = 1850
    Tcalc_start = 1750
    Tcalc_stop = 1550
    Tfor = 1450
    minit = sigma_pebble + sigma_evap
    if (temp > Tcor): #temp > 1800, everything evaporates
        sigma_evap[:] = sigma_evap[:] + sigma_pebble[:]
        sigma_pebble[:] = 0
    elif (temp > Tcalc_start): #1700 < temp < 1800, calcium and forsterite evaporate, corundum condenses
        sigma_pebble[cor_idx] =  sigma_pebble[cor_idx] * (Tcor - temp)/(Tcor - Tcalc_start)
        sigma_evap[cor_idx] =  (minit[cor_idx] - sigma_pebble[cor_idx])
        
        sigma_evap[calc_idx]   = minit[calc_idx] 
        sigma_evap[for_idx]    = minit[for_idx]
        sigma_pebble[calc_idx] = 0
        sigma_pebble[for_idx]  = 0
        
    elif(temp > Tcalc_stop):  # 1500 < temp < 1700, forsterite evaporates, corundum condense
        sigma_pebble[calc_idx] =  sigma_pebble[calc_idx] * (Tcalc_start - temp)/(Tcalc_start - Tcalc_stop)
        sigma_evap[calc_idx] =  (minit[calc_idx] - sigma_pebble[calc_idx])

        sigma_evap[for_idx]   = minit[for_idx]
        sigma_pebble[for_idx] = 0
        
    elif(temp > Tfor):  # 1400 < temp < 1500, forstertite evaporates, calcium and corundum condense
        sigma_pebble[for_idx] =  sigma_pebble[for_idx] * (Tcalc_stop - temp)/(Tcalc_stop - Tfor)
        sigma_evap[for_idx] =  (minit[for_idx] - sigma_pebble[for_idx])

    return (sigma_pebble, sigma_evap)

@numba.njit
def calc_thermal_struc(sigma_gas, sigma_dust, sigma_pebble, sigma_evap, alphas, Omega):
    n   = len(sigma_gas)
    T   = np.empty(n)
    Pmd = np.empty(n)
    cs  = np.empty(n)
    
    for i in range(n): # solve Tmid according to the dust amonut
        # update T_mid and P_mid
        sigma_dust[:,i], sigma_evap[:,i], T[i] = calc_middiskT(sigma_gas[i], sigma_dust[:,i], sigma_evap[:,i], alphas[i], Omega[i])
        sigma_pebble[:,i], sigma_evap[:,i] = evap_cond_pebble(sigma_pebble[:,i], sigma_evap[:,i], T[i])
        cs[i]  = np.sqrt(kB*T[i]/(mu*mH))
        Pmd[i] = sigma_gas[i]*Omega[i]*cs[i]/np.sqrt(2.0*np.pi)
    
    cs2 = np.square(cs)
    return (cs2, T, Pmd, sigma_dust, sigma_evap, sigma_pebble)

@numba.njit
def calc_middiskT(sigma_gas, sigma_dust, sigma_evap, alpha, Omega):
    C0 = 27.0/128*kB/(mu*mH*sb)*np.square(sigma_gas)*Omega
    
    # In a more realistic code, dust amount should be a function of Tmid (mid-disk temperature)
    # Here, we adopt a simplified function as described in calc_opacity, but should be modified in the future.
    
    T0  = (((9*sigma_gas*kB*alpha*Omega)/(8*sb*mu*mH))**(1./3))/10
    Tin = T0
    T1  = T0*100
    f0  = calc_dT(T0, C0, sigma_dust, sigma_evap, sigma_gas)
    f1  = calc_dT(T1, C0, sigma_dust, sigma_evap, sigma_gas)
    
    # make sure bisection range is correct
    count = 0
    while (f0*f1 > 0 and count < 1000):
        T1 *= 1.5
        f1  = calc_dT(T1, C0, sigma_dust, sigma_evap, sigma_gas)

        count += 1

    # bisection search
    eps = 1e-4
    while (np.abs(T1-T0)>eps):
        TA = (T0 + T1)*0.5
        fA = calc_dT(TA, C0, sigma_dust, sigma_evap, sigma_gas)

        if (f0*fA < 0):
            T1 = TA
            f1 = fA
        else:
            T0 = TA
            f0 = fA

    sigma_dust, sigma_evap, kappa = calc_opacity(TA, sigma_dust, sigma_evap, sigma_gas)

    if (TA>1200 and TA<1500):
        print(TA, Tin, f0, f1, "\t", sigma_dust, sigma_evap)
    return (sigma_dust, sigma_evap, TA)

@numba.njit
def calc_dT(T, C0, sigma_dust, sigma_evap, sigma_gas):
    _, _, kpa = calc_opacity(T, sigma_dust, sigma_evap, sigma_gas)
    kpa = kpa * alpha_MRI_hydro(T)
    return pow(kpa*C0, 1.0/3) - T
    
@numba.njit
def calc_opacity(T, sigma_dust, sigma_evap, sigma_gas):
    # kappa_dust * dgratio * Sigma  = kappa_dust * m_dust 
    kpa  = 400
    kgas = 1.6e-3
    sigma_dust, sigma_evap = evap_cond_dust(sigma_dust, sigma_evap, T)
    total_dust = sigma_dust.sum(axis=0)
    dgratio_in = total_dust/sigma_gas
    
    return (sigma_dust, sigma_evap, max(kpa*dgratio_in, kgas))

