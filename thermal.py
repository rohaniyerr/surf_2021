import numpy as np



mH = 1.673534e-27
mu = 2.2 # mean molar mass in disk
kB = 1.380649e-23 #boltzmann constant
sb = 5.6704e-8

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

    cs2 = np.square(cs)
    return (cs2, T, Pmd)

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

def calc_dT(T, dgratio_in, C0):
    kpa = calc_opacity(T, dgratio_in)
    return pow(kpa*C0, 1.0/3) - T
    
def calc_opacity(T, dgratio_in):
    # kappa_dust * dgratio * Sigma  = kappa_dust * m_dust 
    kpa  = 0.5
    kgas = 1.6e-3

    # Tcor and Tfor correspond to the evaporation temperature of corundum (Al2O3) and forsterite (MgSiO4)
    # This part should be modified later.
    Tcor = 1800
    Tfor = 1400
    if (T > Tcor):
        dgratio = 0.
    elif (T > Tfor):
        dgratio = dgratio_in * (Tcor-T)/(Tcor-Tfor)
    else:
        dgratio = dgratio_in
    
    return max(kpa*dgratio, kgas)
