import numpy as np



mH = 1.673534 * 10**-27
mu = 2.2 # mean molar mass in disk
kB = 1.380649 * 10**-23 #boltzmann constant
sb = 5.6704*10**-8


def calc_thermal_struc(sigma_gas, sigma_dust, alphas, Omega):
    T = np.empty(len(sigma_gas))
    Pmd = np.empty(len(sigma_gas))
    cs = np.empty(len(sigma_gas))
    for i in range(len(sigma_gas)):
    # solve Tmid according to the dust amonut
        dgratio = sigma_dust[i]/sigma_gas[i]

        # update T_mid and P_mid
        C0 = 27.0/128*alphas[i]*kB/(mu*mH*sb)*np.square(sigma_gas[i])*Omega[i]
        kpa = calc_opacity(dgratio)

        T[i] = pow(kpa*C0, 1.0/3)
        cs[i]  = np.sqrt(kB*T[i]/(mu*mH))
        Pmd[i] = sigma_gas[i]*Omega[i]*cs[i]/np.sqrt(2.0*np.pi)
    cs2 = np.square(cs)
    return (cs2, T, Pmd)


def calc_opacity(dgratio):
    # kappa_dust * dgratio * Sigma  = kappa_dust * m_dust 
    kpa  = 0.5
    kgas = 1.6e-3
    
    return max(kpa*dgratio, kgas)