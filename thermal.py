import numpy as np
import numba


mH = 1.673534e-27
mu = 2.2 # mean molar mass in disk
kB = 1.380649e-23 #boltzmann constant
sb = 5.6704e-8
G = 6.6738 * 10**-11 #gravitational constant
Ms = 1.9886 * 10**30 #solar mass
Tcor = 1800
Tfor = 1400
Tcalc = 1700
# cor_ratio = .1
# calc_ratio = .1
# for_ratio = .8

# @numba.njit
# def evaporate_dust(sigma_dust, T):
#     sigma_evap = np.zeros(len(sigma_dust))
#     for idx, temp in enumerate(T):
#         if (temp > Tcor): #temp > 1800, everything evaporates
#             sigma_evap[idx] = sigma_evap[idx] + sigma_dust[idx]
#             sigma_dust[idx] = 0
#         elif (temp > Tcalc): #1700 < temp < 1800, calcium and forsterite evaporate
#             sigma_evap[idx] = sigma_evap[idx] + (calc_ratio + for_ratio) * sigma_dust[idx]*(Tcor-temp)/(Tcor-Tcalc)
#             sigma_dust[idx] = cor_ratio*sigma_dust[idx]*(Tcor-temp)/(Tcor-Tcalc)
#         elif(temp > Tfor):  # 1400 < temp < 1700, only forsterite evaporates
#             sigma_evap[idx] = sigma_evap[idx] + for_ratio * sigma_dust[idx] * (Tcalc-temp)/(Tcalc-Tfor)
#             sigma_dust[idx] = (calc_ratio + cor_ratio) * sigma_dust[idx] * (Tcalc-temp)/(Tcalc-Tfor)
#     return (sigma_evap, sigma_dust)

# @numba.njit
# def condense_gas(sigma_evap, sigma_dust, T):
#     for idx, temp in enumerate(T):
#         if (temp <= Tfor): #temp < 1400, everything condenses
#             sigma_dust[idx] = sigma_dust[idx] + sigma_evap[idx]
#             sigma_evap[idx] = 0
#         elif (temp <= Tcalc): # 1400 < temp < 1700, calcium and corundum condense
#             sigma_dust[idx] = sigma_dust[idx] + (calc_ratio + cor_ratio) * sigma_evap[idx] * (Tcalc-temp)/(Tcalc-Tfor)
#             sigma_evap[idx] = for_ratio * sigma_evap[idx] * (Tcalc-temp)/(Tcalc-Tfor)
#         elif (temp <= Tcor): # 1700 < temp < 1800, only corundum condenses
#             sigma_dust[idx] = sigma_dust[idx] + cor_ratio * sigma_evap[idx] * (Tcor-temp)/(Tcor-Tcalc)
#             sigma_evap[idx] = (calc_ratio + for_ratio) * sigma_evap[idx] * (Tcor-temp)/(Tcor-Tcalc) 
#     return sigma_dust


def calc_thermal_struc(sigma_gas, sigma_dust, alphas, Omega):
    T = np.empty(len(sigma_gas))
    Pmd = np.empty(len(sigma_gas))
    cs = np.empty(len(sigma_gas))
    
    for i in range(len(sigma_gas)):
    # solve Tmid according to the dust amonut
        dgratio = sigma_dust[i]/sigma_gas[i]
        # update T_mid and P_mid
        T[i] = calc_middiskT(dgratio, sigma_gas[i], alphas[i], Omega[i])
        cs[i]  = np.sqrt(kB*T[i]/(mu*mH))
        Pmd[i] = sigma_gas[i]*Omega[i]*cs[i]/np.sqrt(2.0*np.pi)
#     sigma_evap, sigma_dust = evaporate_dust(sigma_dust, T)
#     sigma_dust = condense_gas(sigma_evap, sigma_dust, T)
    cs2 = np.square(cs)
    return (cs2, T, Pmd)

@numba.njit
def calc_middiskT(dgratio, sigma_gas, alpha, Omega):
    C0 = 27.0/128*alpha*kB/(mu*mH*sb)*np.square(sigma_gas)*Omega
    
    # In a more realistic code, dust amount should be a function of Tmid (mid-disk temperature)
    # Here, we adopt a simplified function as described in calc_opacity, but should be modified in the future.
    
    T0  = ((9*sigma_gas*kB*alpha*Omega)/(8*sb*mu*mH))**(1./3)
    Tin = T0
    T1  = T0*10
    f0  = calc_dT(T0, dgratio, C0)
    f1  = calc_dT(T1, dgratio, C0) 
    eps = 1e-2
    while (np.abs(T1-T0)>eps):
        TA = (T0 + T1)*0.5
        fA = calc_dT(TA, dgratio, C0)

        if (f0*fA < 0):
            T1 = TA
            f1 = fA
        else:
            T0 = TA
            f0 = fA

    kappa = calc_opacity(TA, dgratio)
    return TA


@numba.njit
def calc_dT(T, dgratio_in, C0):
    kpa = calc_opacity(T, dgratio_in)
    return pow(kpa*C0, 1.0/3) - T
    
@numba.njit
def calc_opacity(T, dgratio_in):
    # kappa_dust * dgratio * Sigma  = kappa_dust * m_dust 
    kpa  = 0.5
    kgas = 1.6e-3

    # Tcor and Tfor correspond to the evaporation temperature of corundum (Al2O3) and forsterite (MgSiO4)
    # This part should be modified later.
    if (T > Tcor):
        dgratio = 0.
    elif (T > Tcalc):
        dgratio = dgratio_in * (Tcor-T)/(Tcor-Tcalc)
    elif (T > Tfor):
        dgratio = dgratio_in * (Tcalc-T)/(Tcalc-Tfor)
    else:
        dgratio = dgratio_in
    
    return max(kpa*dgratio, kgas)

