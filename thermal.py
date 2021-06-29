def calc_thermal_struc(alpha):
    #pragma omp parallel for schedule(dynamic,1)
    for i in range(len(sigma)):
	# solve Tmid according to the dust amonut
	dgratio = sigma_dust[i]/sigma[i]
            
	# update T_mid and P_mid
	C0 = 27.0/128*alpha*kB/(mu*mH*sb)*square(sig[i])*Omeg[i]
	kpa = calc_opacity(Tmd[i], dgratio, s_max[i])
            
	T[i] = pow(kpa*C0, 1.0/3)
	cs  = sqrt(kB*Tmd[i]/(mu*mH))
	Pmd = sig[i]*Omeg[i]*cs/sqrt(2.0*pi)
    }
}
