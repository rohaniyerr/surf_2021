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
def evap_cond(sigma_d_in, sigma_ev_in, temp):
    minit = sigma_d_in + sigma_ev_in
    
    if (temp > Tcor): #temp > 1800, everything evaporates
        sigma_evap = minit
        sigma_dust = 0
    elif (temp > Tcalc_start): #1700 < temp < 1800, calcium and forsterite evaporate, corundum condenses
        sigma_evap = cor_ratio * minit * (temp - Tcalc_start)/(Tcor - Tcalc_start) + (for_ratio + calc_ratio)*minit
        sigma_dust = minit - sigma_evap
    elif(temp > Tcalc_stop):  # 1500 < temp < 1700, forsterite evaporates, corundum condense
        sigma_evap = (calc_ratio) * minit * (temp-Tcalc_stop)/(Tcalc_start-Tcalc_stop) + (for_ratio) * minit
        sigma_dust = minit - sigma_evap
    elif(temp > Tfor):  # 1400 < temp < 1500, forstertite evaporates, calcium and corundum condense
        sigma_evap = (for_ratio) * minit * (temp - Tfor)/(Tcalc_stop - Tfor)
        sigma_dust = minit - sigma_evap
    else: #temp < 1400, everything condenses
        sigma_dust = minit
        sigma_evap = 0

    return (sigma_dust, sigma_evap)


def calc_thermal_struc(sigma_gas, sigma_dust, sigma_evap, alphas, Omega):
    T = np.empty(len(sigma_gas))
    Pmd = np.empty(len(sigma_gas))
    cs = np.empty(len(sigma_gas))
    
    for i in range(len(sigma_gas)):
    # solve Tmid according to the dust amonut
#         dgratio = (sigma_dust[i] + sigma_evap[i]) / sigma_gas[i]
        # update T_mid and P_mid
        T[i] = calc_middiskT(sigma_gas[i], sigma_dust[i], sigma_evap[i], alphas[i], Omega[i])
        cs[i]  = np.sqrt(kB*T[i]/(mu*mH))
        Pmd[i] = sigma_gas[i]*Omega[i]*cs[i]/np.sqrt(2.0*np.pi)
    cs2 = np.square(cs)
    return (cs2, T, Pmd)

@numba.njit
def calc_middiskT(sigma_gas, sigma_dust, sigma_evap, alpha, Omega):
    C0 = 27.0/128*kB/(mu*mH*sb)*np.square(sigma_gas)*Omega  # no alpha here
    
    # In a more realistic code, dust amount should be a function of Tmid (mid-disk temperature)
    # Here, we adopt a simplified function as described in calc_opacity, but should be modified in the future.
    
    T0  = ((9*sigma_gas*kB*alpha*Omega)/(8*sb*mu*mH))**(1./3)
    Tin = T0
    T1  = T0*10
    f0  = calc_dT(T0, C0, sigma_dust, sigma_evap, sigma_gas)
    f1  = calc_dT(T1, C0, sigma_dust, sigma_evap, sigma_gas) 
    eps = 1e-2
    while (np.abs(T1-T0)>eps):
        TA = (T0 + T1)*0.5
        fA = calc_dT(TA, C0, sigma_dust, sigma_evap, sigma_gas)

        if (f0*fA < 0):
            T1 = TA
            f1 = fA
        else:
            T0 = TA
            f0 = fA

    kappa = calc_opacity(TA, sigma_dust, sigma_evap, sigma_gas)
    return TA


@numba.njit
def calc_dT(T, C0, sigma_dust, sigma_evap, sigma_gas):
    kpa = calc_opacity(T, sigma_dust, sigma_evap, sigma_gas) * alpha_MRI_hydro(T)
    return pow(kpa*C0, 1.0/3) - T
    
@numba.njit
def calc_opacity(T, sigma_dust, sigma_evap, sigma_gas):
    # kappa_dust * dgratio * Sigma  = kappa_dust * m_dust 
    kpa  = 0.5
    kgas = 1.6e-3
    sigma_dust, sigma_evap = evap_cond(sigma_dust, sigma_evap, T)
    dgratio_in = sigma_dust/sigma_gas

    #print(dgratio_in, sigma_dust+sigma_evap)
    return max(kpa*dgratio_in, kgas)

#     # Tcor and Tfor correspond to the evaporation temperature of corundum (Al2O3) and forsterite (MgSiO4)
#     # This part should be modified later.
#     if (T > Tcor):
#         dgratio = 0.
#     elif (T > Tcalc):
#         dgratio = dgratio_in * (Tcor-T)/(Tcor-Tcalc)
#     elif (T > Tfor):
#         dgratio = dgratio_in * (Tcalc-T)/(Tcalc-Tfor)
#     else:
#         dgratio = dgratio_in
    
    

