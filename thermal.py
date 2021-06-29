import numpy as np

def calc_thermal_struc(alpha):
    #pragma omp parallel for schedule(dynamic,1)
    for i in range(len(sigma)):
	# solve Tmid according to the dust amonut
	dgratio = sigma_dust[i]/sigma[i]
            
	# update T_mid and P_mid
	C0 = 27.0/128*alpha[i]*kB/(mu*mH*sb)*square(sigma[i])*Omega[i]
	kpa = calc_opacity(Tmd[i], dgratio, s_max[i])
        
	T[i] = pow(kpa*C0, 1.0/3)
	cs  = sqrt(kB*T[i]/(mu*mH))
	Pmd = sig[i]*Omega[i]*cs/sqrt(2.0*np.pi)


def calc_opacity(T, dgratio):
    # kappa_dust * dgratio * Sigma  = kappa_dust * m_dust 
    kpa  = 0.5
    kgas = 1.6e-3
    
    return max(kpa*dgratio, kgas)

