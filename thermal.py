import numpy as np
import numba


mH = 1.673534e-27
mu = 2.2 # mean molar mass in disk
kB = 1.380649e-23 #boltzmann constant
sb = 5.6704e-8
G = 6.6738 * 10**-11 #gravitational constant
Ms = 1.9886 * 10**30 #solar mass

Tcor = 1800
Tcalc_start = 1700
Tcalc_stop = 1500
Tfor = 1400

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
def evap_cond(sigma_dust, sigma_evap, sigma_dust_fin, sigma_evap_fin, temp, idx):
    minit = sigma_dust + sigma_evap
    cor_idx = 0
    calc_idx = 1
    for_idx = 2
    if (temp > Tcor): #temp > 1800, everything evaporates
        sigma_evap = sigma_evap + sigma_dust
        sigma_dust = 0
        sigma_evap_fin[:,idx] = sigma_evap
        sigma_dust_fin[:,idx] = sigma_dust
    elif (temp > Tcalc_start): #1700 < temp < 1800, calcium and forsterite evaporate, corundum condenses
        sigma_evap = cor_ratio * minit * (temp - Tcalc_start)/(Tcor - Tcalc_start)
        sigma_dust = cor_ratio * minit * (Tcor - temp)/(Tcor - Tcalc_start) 
        sigma_evap_fin[cor_idx,idx] = sigma_evap
        sigma_dust_fin[cor_idx,idx] = sigma_dust
    elif(temp > Tcalc_stop):  # 1500 < temp < 1700, forsterite evaporates, corundum condense
        sigma_evap = (calc_ratio) * minit * (temp - Tcalc_stop)/(Tcalc_start - Tcalc_stop)
        sigma_dust = (calc_ratio) * minit * (Tcalc_start - temp)/(Tcalc_start - Tcalc_stop)
        sigma_evap_fin[calc_idx,idx] = sigma_evap
        sigma_dust_fin[calc_idx,idx] = sigma_dust
    elif(temp > Tfor):  # 1400 < temp < 1500, forstertite evaporates, calcium and corundum condense
        sigma_evap = (for_ratio) * minit * (temp - Tfor)/(Tcalc_stop - Tfor)
        sigma_dust= (for_ratio) * minit * (Tcalc_stop - temp)/(Tcalc_stop - Tfor)
        sigma_evap_fin[for_idx,idx] = sigma_evap
        sigma_dust_fin[for_idx,idx] = sigma_dust
    else: #temp < 1400, everything condenses
        sigma_dust = sigma_dust + sigma_evap
        sigma_evap = 0
        sigma_evap_fin[:,idx] = sigma_evap
        sigma_dust_fin[:,idx] = sigma_dust
    return (sigma_dust, sigma_evap, sigma_dust_fin, sigma_evap_fin)


def calc_thermal_struc(sigma_gas, sigma_dust, sigma_evap, alphas, Omega):
    n = len(sigma_gas)
    T = np.empty(n)
    Pmd = np.empty(n)
    cs = np.empty(n)
    sigma_dust_fin = np.zeros((3,n))
    sigma_evap_fin = np.zeros((3,n))
    total_dust = sigma_dust.sum(axis=0)
    total_evap = sigma_evap.sum(axis=0)
    for i in range(n): # solve Tmid according to the dust amonut
        # update T_mid and P_mid
#             sigma_dust_fin[i], sigma_evap_fin[i], 
        sigma_dust_fin, sigma_evap_fin, T[i] = calc_middiskT(sigma_gas[i], total_dust[i], total_evap[i], sigma_dust_fin, sigma_evap_fin, alphas[i], Omega[i], i)
        cs[i]  = np.sqrt(kB*T[i]/(mu*mH))
        Pmd[i] = sigma_gas[i]*Omega[i]*cs[i]/np.sqrt(2.0*np.pi)
    cs2 = np.square(cs)
    return (cs2, T, Pmd, sigma_dust_fin, sigma_evap_fin)

@numba.njit
def calc_middiskT(sigma_gas, sigma_dust, sigma_evap, sigma_dust_fin, sigma_evap_fin, alpha, Omega, idx):
    C0 = 27.0/128*kB/(mu*mH*sb)*np.square(sigma_gas)*Omega
    
    # In a more realistic code, dust amount should be a function of Tmid (mid-disk temperature)
    # Here, we adopt a simplified function as described in calc_opacity, but should be modified in the future.
    
    T0  = (((9*sigma_gas*kB*alpha*Omega)/(8*sb*mu*mH))**(1./3))/10
    Tin = T0
    T1  = T0*10
    f0  = calc_dT(T0, C0, sigma_dust, sigma_evap, sigma_gas, sigma_dust_fin, sigma_evap_fin, idx)
    f1  = calc_dT(T1, C0, sigma_dust, sigma_evap, sigma_gas, sigma_dust_fin, sigma_evap_fin, idx) 
    eps = 1e-2
    while (np.abs(T1-T0)>eps):
        TA = (T0 + T1)*0.5
        fA = calc_dT(TA, C0, sigma_dust, sigma_evap, sigma_gas, sigma_dust_fin, sigma_evap_fin, idx)

        if (f0*fA < 0):
            T1 = TA
            f1 = fA
        else:
            T0 = TA
            f0 = fA

    _, _, kappa, sigma_dust_fin, sigma_evap_fin = calc_opacity(TA, sigma_dust, sigma_evap, sigma_gas,sigma_dust_fin, sigma_evap_fin, idx)
    print(sigma_dust_fin)
    return (sigma_dust_fin, sigma_evap_fin, TA)

@numba.njit
def calc_dT(T, C0, sigma_dust, sigma_evap, sigma_gas,sigma_dust_fin, sigma_evap_fin, idx):
    _, _, kpa,_ , _ = calc_opacity(T, sigma_dust, sigma_evap, sigma_gas,sigma_dust_fin, sigma_evap_fin, idx)
    kpa = kpa * alpha_MRI_hydro(T)
    return pow(kpa*C0, 1.0/3) - T
    
@numba.njit
def calc_opacity(T, sigma_dust, sigma_evap, sigma_gas, sigma_dust_fin, sigma_evap_fin, idx):
    # kappa_dust * dgratio * Sigma  = kappa_dust * m_dust 
    kpa  = 0.5
    kgas = 1.6e-3
    sigma_dust, sigma_evap, sigma_dust_fin, sigma_evap_fin = evap_cond(sigma_dust, sigma_evap, sigma_dust_fin, sigma_evap_fin, T, idx)
    dgratio_in = sigma_dust/sigma_gas
    return (sigma_dust, sigma_evap, max(kpa*dgratio_in, kgas), sigma_dust_fin, sigma_evap_fin)

    


    
# @numba.njit
# def evap_cond(sigma_dust, sigma_evap, temp):
#     minit = sigma_dust + sigma_evap
#     cor_idx = 1
#     calc_idx = 2
#     fors_idx = 3
#     if (temp > Tcor): #temp > 1800, everything evaporates
#         sigma_evap = sigma_evap + sigma_dust
#         sigma_dust = 0
#     elif (temp > Tcalc_start): #1700 < temp < 1800, calcium and forsterite evaporate, corundum condenses
#         sigma_evap[cor_idx,:] = cor_ratio * minit * (temp - Tcalc_start)/(Tcor - Tcalc_start) + (for_ratio + calc_ratio)*minit
#         sigma_dust = cor_ratio * minit * (Tcor - temp)/(Tcor - Tcalc_start) 
#     elif(temp > Tcalc_stop):  # 1500 < temp < 1700, forsterite evaporates, corundum condense
#         sigma_evap = (calc_ratio) * minit * (temp - Tcalc_stop)/(Tcalc_start - Tcalc_stop) + (for_ratio) * minit
#         sigma_dust = (calc_ratio) * minit * (Tcalc_start - temp)/(Tcalc_start - Tcalc_stop) + (cor_ratio) * minit
#     elif(temp > Tfor):  # 1400 < temp < 1500, forstertite evaporates, calcium and corundum condense
#         sigma_evap = (for_ratio) * minit * (temp - Tfor)/(Tcalc_stop - Tfor)
#         sigma_dust = (for_ratio) * minit * (Tcalc_stop - temp)/(Tcalc_stop - Tfor) + (calc_ratio + cor_ratio)*minit
#     else: #temp < 1400, everything condenses
#         sigma_dust = sigma_dust + sigma_evap
#         sigma_evap = 0
#     return (sigma_dust, sigma_evap)