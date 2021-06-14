/*
  grid.cpp
  ... a grid for radial modeling.
  
  includes 1. physical parameters (r, T, P)
           2. composition (n_dust, n_migrator, n_planetesimal)
*/

#include "grid.h"

radial::radial(int Ngrid, double r0, double r1){
    /* set-up the grid */
    if (r0 == 0){
        cerr << "The inner boundary of 'radial' cannot be set to 0." << endl; 
        exit(7);
    }
    dist.resize(Ngrid);
    Omeg.resize(Ngrid);
    for (int i=0; i<Ngrid; i++){
        dist[i] = (r0 + (r1-r0)*i/(Ngrid-1)) * AU;
        Omeg[i] = sqrt(G*Ms/(dist[i]*dist[i]*dist[i]));
    }

    /* staggered grid */
    dist_stg.resize(Ngrid+1);
    Omeg_stg.resize(Ngrid+1);
    for (int i=0; i<(Ngrid+1); i++){
        dist_stg[i] = (dist[i-1]+dist[i])/2.0;
        Omeg_stg[i] = sqrt(G*Ms/(dist_stg[i]*dist_stg[i]*dist_stg[i]));
    }
                       
    
    Tmd.resize(0.0, Ngrid);
    sig.resize(0.0, Ngrid);
    Pmd.resize(0.0, Ngrid);
    no_plts.resize(0.0, Ngrid);
    dm_ev.resize(0.0, Ngrid);
    s_max.resize(0, Ngrid);
    Rplts.resize(0, Ngrid);
    RaccB.resize(0, Ngrid);
    RaccH.resize(0, Ngrid);
    
    dmigr.resize(0, Ngrid);
    dmout.resize(0, Ngrid);
    dmin.resize(0, Ngrid);
    vdrift.resize(0, Ngrid);
    Steta.resize (0, Ngrid);
    dPdr.resize(  0, Ngrid);
    
}
void radial::set_init_density(double sigma_in, double sigma_max, double Rplts_in, string spec_list_in, string comp_list){
    /* set composition */
    spec_list = spec_list_in;
    MoleculeData_G   list_ref(spec_list, 298.16, 1.0);
    cout << "comp: " << comp_list << endl;
    initcomp         start_comp(comp_list, list_ref);
    massBalance      matrix(list_ref);
    
    massm = matrix.get_massBalance();
    
    tensor1d<double> n_init = start_comp.getmole_t();
    n_mass = list_ref.getmolarmass_t();
    double unit_mass = n_mass*n_init;
    n_init /= unit_mass;
    
    /* set density */
    double sigma_min = 1e2;
    for (int i=0; i<(int)sig.size(); i++){
        sig[i] = sigma_in*(1.0*AU)/dist[i];
        
        if (sig[i]>sigma_max){ sig[i] = sigma_max; }
        if (dist[i]/AU > 15 ){ sig[i] = sigma_min; }
    }
    
    /* prepare composition */
    int  numofcomp = n_init.size();
    int  numofel   = massm.mcols();
    ngas.resize(0,  sig.size(), numofcomp);
    ndust.resize(0, sig.size(), numofcomp);
    nmigr.resize(0, sig.size(), numofcomp);
    nplts.resize(0, sig.size(), numofcomp);
    del_ev.resize(0, sig.size(), numofel);
    for (int i=0; i<(int)sig.size(); i++){ ngas.setrowtensor(i, n_init*sig[i]); }
    
    /* display disk mass */
    double Mdisk = 0.0, dr = dist[1]-dist[0];
    for (int i=0; i<(int)sig.size(); i++){ Mdisk += (2*pi*dist[i]*dr)*sig[i]; }
    cout << "Mdisk = " << Mdisk/Ms << endl;
    
    /* set initial isotopic composition */
    iso_gasd.resize(0, sig.size(), massm.mcols());
    iso_migr.resize(0, sig.size(), massm.mcols());
    iso_plts.resize(0, sig.size(), massm.mcols());
    for (int i=0; i<(int)sig.size(); i++){
        if (dist[i]/AU > 7) { iso_gasd[i][1] = 1; }  /* O2 */
        if (dist[i]/AU > 4) { iso_gasd[i][4] = 1; }  /* Al */
        if (dist[i]/AU > 4) { iso_gasd[i][6] = 1; }  /* Ca */
        
        /* planetesimal size */
        Rplts[i] = Rplts_in;
    }  
    
}
void radial::read_file(string filename){
    ifstream fin(filename);
    if (!fin){ cout << "Run the program from t=0" << endl; exit(2); }

    
    for (int i=0; i<(int)sig.size(); i++){
        double temp;
        fin >> temp >> sig[i] >> temp >> temp >> no_plts[i] >> Rplts[i] >> RaccB[i] >> RaccH[i];
        fin >> dm_ev[i] >> Tmd[i] >> Pmd[i] >> temp;
        
        for (int j=0; j<ndust.mcols(); j++){ fin >> ngas[i][j]; }
        
        for (int k=0; k<massm.mcols(); k++){ fin >> temp; }
        for (int k=0; k<massm.mcols(); k++){ fin >> temp; }
        for (int k=0; k<massm.mcols(); k++){ fin >> temp; }
        for (int k=0; k<massm.mcols(); k++){ fin >> del_ev[i][k]; }
        
        /* isotope */
        for (int k=0; k<iso_gasd.mcols(); k++){ fin >> iso_gasd[i][k]; }
        for (int k=0; k<iso_migr.mcols(); k++){ fin >> iso_migr[i][k]; }
        for (int k=0; k<iso_plts.mcols(); k++){ fin >> iso_plts[i][k]; }
        
        fin >> temp >> temp >> temp >> temp;
        fin >> temp >> temp >> temp >> temp;
        
        for (int j=0; j<ndust.mcols(); j++){ fin >> nmigr[i][j]; }

    }

    fin.close();
    return;

}

/*--------------------------calc_thermal_struc------------------------------*/
/* 
   two types of function to calculate are prepared.

   1. calc_thermal_Gibbs_struc is the an accurate way to obtain T_mid and dust amount self-consistently
   2. calc_thermal_struc is a simplified version to calc T_mid based on the given dust amonut

 */
void radial::calc_thermal_struc(double alpha){
    //#pragma omp parallel for schedule(dynamic,1)
    for (int i=0; i<(int)Tmd.size(); i++){
        if (Tmd[i] < 400){
            /* solve Tmid according to the dust amonut  */
            tensor1d<double> vgas = ngas.rowtensor(i);
            tensor1d<double> vsol = ndust.rowtensor(i);
            double dgratio = (vsol*n_mass)/(vgas*n_mass);
            
            /* update T_mid and P_mid */
            double C0 = 27.0/128*alpha*kB/(mu*mH*sb)*square(sig[i])*Omeg[i];
            double kpa = calc_opacity(Tmd[i], dgratio, s_max[i]);
            
            // cout << i << "\t" << Tmd[i] << "\t" << pow(kpa*C0, 1.0/3) << endl;
            Tmd[i] = pow(kpa*C0, 1.0/3);
            double cs  = sqrt(kB*Tmd[i]/(mu*mH));
            Pmd[i] = sig[i]*Omeg[i]*cs/sqrt(2.0*pi);
        }
    }
}
void radial::calc_thermal_Gibbs_struc(double alpha){
    clock_t begin = clock();  
    remove_unresolvable();
      
#pragma omp parallel for schedule(dynamic,1)
    for (int i=0; i<(int)Tmd.size(); i++){
        /* solve Tmid and dust amount self-consistently 
           ... first, obtain composition in grid i */
        tensor1d<double> vtot = ngas.rowtensor(i) + ndust.rowtensor(i);
        
        //double C0 = 27.0/64*alpha*kB/(mu*mH*sb)*square(sig[i])*Omeg[i];
        double C0 = 27.0/128*alpha*kB/(mu*mH*sb)*square(sig[i])*Omeg[i];
        double Tp = Tmd[i];
        double P_over_sqrtT = sig[i]*Omeg[i]*sqrt(kB/(2.0*pi*mu*mH));
        
        double Tc;
        tensor1d<double> vgas, vdust;
        if (Tp < 300 or Tp > 2500){
            tie(vgas, vdust, Tc) = solve_middiskT(vtot, Tp, P_over_sqrtT, C0, s_max[i]); }
        else{
            tie(vgas, vdust, Tc) = solve_vertical(vtot, Tp, sig[i], Omeg[i], alpha);
        }
        
        double cs = sqrt(kB*Tc/(mu*mH));
        
#pragma omp critical
        {
            Tmd[i] = Tc;
            Pmd[i] = sig[i]*Omeg[i]*cs/sqrt(2.0*pi);
            ngas.setrowtensor( i, vgas);
            ndust.setrowtensor(i, vdust);
            
            /*
              check the existence of condensation front
              ... when enstatie starts to coagulate near the mid-plane, 
            */
            if (vdust[16]==0 and vdust[22]==0){ s_max[i] = 0; }
            else                              { s_max[i] = 0; }
            
        }
        
        tensor1d<double> vmigr = nmigr.rowtensor(i);
        double pgratio = (vmigr*n_mass)/(vgas*n_mass); 
        //cout << dist[i]/AU << "\t" << Tmd[i] << " K\t" << Pmd[i] << " Pa\t" << sig[i] << " kg/m3\t";
        //cout << pgratio << "/" << no_plts[i] << " km \n"; 
    }
    clock_t end = clock();
    //cout << "therm-Gibbs: " << (double)(end-begin)/CLOCKS_PER_SEC << endl;
}
tuple<tensor1d<double>, tensor1d<double>, double> radial::solve_middiskT(tensor1d<double> vtot, double Tref,
                                                                         double P_over_sqrtT, double C0, int ev){
    /* bisection search
     ... because GFE is required in each step in the loop, this calc is time consuming.
     in order to minimize the no. of iteration, T range is determined from the previous temperature.
     
     The exception is the very first timestep. Tmd is set to 0 across the disk, so we adopt
     dT_range = 2000 to search within entire possible T range. */
    double dT_range = 10, eps = 1;
    if     (Tref < 1e-3){ dT_range = 2000; eps = 1e-4; }
    else if(Tref < 1e1) { eps = 1e-3; }
    double T0 = maxv(Tref-dT_range, 0.01), T1 = Tref+dT_range, TA;
    
    /* search between T0 and T1 
       ... Pressure is also necessary. P = P_over_sqrtT*sqrt(T) */
    double f0, f1, fA;  tensor1d<double> vgas, vsol;
    tie(vgas, vsol, f0) = delta_Tcube(vtot, T0, P_over_sqrtT*sqrt(T0), C0, ev);
    tie(vgas, vsol, f1) = delta_Tcube(vtot, T1, P_over_sqrtT*sqrt(T1), C0, ev);
    
    /* When a correct set of (Tmid, dust amount) cannot be found between (T0,T1),
       the search range will be expanded.  */
    if (f0*f1 > 0){ T0 = 0.1; }
    while (f0*f1 > 0){
        T0 -= 10; T1 += 1000;
        if (T0 < 0){ T0 = 0.1; }
        tie(vgas, vsol, f0) = delta_Tcube(vtot, T0, P_over_sqrtT*sqrt(T0), C0, ev);
        tie(vgas, vsol, f1) = delta_Tcube(vtot, T1, P_over_sqrtT*sqrt(T1), C0, ev);
    }
    
    /* bisection search
       --- find a set of (Tmid, dust amount) that satisfies (radiative cooling) = (viscous dissipation) 
       --- 9/8 nu Sigma Omega^2 = Stefan-Boltzmann * (T_surf)^4.
       
       --- In order to save comupational time, we skip the GFE for temperature below 1000 K
       --- because it is known that the dust amount does not vary.*/
    while (abs(T1-T0) > eps && f0*f1 < 0){
        TA = (T0+T1)/2.0;
        if (TA < 300 or TA > 2000){
            double dgratio = (vsol*n_mass)/(vgas*n_mass);
            double kpa = calc_opacity(TA, dgratio, ev);
            fA = TA*TA*TA-kpa*C0;
        }else{
            tie(vgas, vsol, fA) = delta_Tcube(vtot, TA, P_over_sqrtT*sqrt(TA), C0, ev);
        }
        
        if (f0*fA > 0){ T0 = TA;  f0 = fA; }
        else          { T1 = TA;  f1 = fA; }
    }
    if (TA < 300 or TA > 2000){ tie(vgas, vsol, fA) = delta_Tcube(vtot, TA, P_over_sqrtT*sqrt(TA), C0, ev); }
    
    return forward_as_tuple(vgas, vsol, (T0+T1)/2.0);
}
tuple<tensor1d<double>, tensor1d<double>, double> radial::delta_Tcube(tensor1d<double> vcomp,
                                                                      double T, double P, double C0, int ev){
    /*
      return gas&dust compositions, and T^3 - C0*kappa, which is zero at equilibrium state
    */
    tensor1d<double> vgas, vsol;
    bool converge;
    
    /* repeat the minimization until it converges but only for T > 0.1.
       If T < 0.1, the effect of reaction should be minimum, so gibbsminCG is not performed in this case...
    */
    if (T > 0.1){
        for (int iter=0; iter<2; iter++){
            MoleculeData_G gibbs_list(spec_list, T, P);
            gibbsminCG     min(gibbs_list, vcomp);
            vgas = min.getngas_or_solid(0);
            vsol = min.getngas_or_solid(2);
            converge = min.convergence();
            
            if (converge){ break; }
            T += 10*((double)rand()/RAND_MAX*2-1);
            // cout << T << " K \t" << P << "Pa " << iter+1 << endl;
        }
    }else{
        /* if T < 10, just split vcomp to vgas/vsol according to information in gibbs_list */
        vgas.resize(0.0, vcomp.size());
        vsol.resize(0.0, vcomp.size());
        
        MoleculeData_G gibbs_list(spec_list, T, P);
        for (int j=0; j<(int)vcomp.size(); j++){
            if (gibbs_list[j].getphase() == 0){ vgas[j] = vcomp[j]; }
            if (gibbs_list[j].getphase() == 2){ vsol[j] = vcomp[j]; }
        }
    }
    
    double mgas = vgas*n_mass, mdust = vsol*n_mass, dgratio = mdust/mgas;
    double kpa = calc_opacity(T, dgratio, ev);
    // cout << "  " <<  T*T*T - kpa*C0 << "\t kp = " << kpa << " @" << T << " K" << " min " << min.getGbest() << endl;
    
    return forward_as_tuple(vgas, vsol, T*T*T - kpa*C0);
}
double radial::calc_opacity(double T, double dgratio, int ev){
    /* kappa_dust * dgratio * Sigma  = kappa_dust * m_dust */
    double kpa = (ev)? opacity_ave[1] : opacity_ave[0];    
    double kgas = 1.6e-3;
    if (T < 10){ dgratio = 5.6e-3; }
    
    return maxv(kpa*dgratio, kgas);
}

/*--------------------------------------------------------------------------------------
  migrator (pebble)
*/
void radial::migrator_equili(string spec_list, double alpha0, double p_part){
    /* 
       --- General :
       Maintain chemical equilibrium between migrator and gas+dust
       
       --- 
       calc_thermal_Gibbs() calculates thermal structure to be consistent with dust amount,
       but migrators are not considered b/c they are too big to affect the opacity.
      
    */
    tensor1d<double> vzero(0.0, ndust.mcols());
    
#pragma omp parallel for schedule(dynamic,1)
    for (int i=0; i<ndust.nrows(); i++){
        /* 
           The effect of migrator evaporation/re-condensation is minimal further away from the Sun,
           so their effects are ignored.
        */
        if (dist[i]/AU > 10){ continue; }
        
        /*
          --- Step 1:
          Gibbs free energy minimiziation is performed for the `gas` + `migrator` compositions first.
          
          Consider the equilibrium between gas and migrator phases (NOT dust phase).
          `dm_ev` records the change in migrator dust amont.
         */
        const tensor1d<double> vgas  = ngas.rowtensor(i);
        tensor1d<double>       vmigr = nmigr.rowtensor(i);
        tensor1d<double>       vtot  = vgas + vmigr;
        tensor1d<double>       vgas_aft, vmigr_aft;
        
        /* repeat the minimization until it converges */
        double Ttry = Tmd[i];
        for (int iter=0; iter<2; iter++){
            MoleculeData_G   gibbs_list(spec_list, Ttry, Pmd[i]);
            gibbsminCG       min(gibbs_list, vtot);
            vgas_aft  = min.getngas_or_solid(0);
            vmigr_aft = min.getngas_or_solid(2);
            bool converge = min.convergence();
            
            if (converge){ break; }
            Ttry += 10*((double)rand()/RAND_MAX*2-1);
            //cout << Ttry << " K \t" << Pmd[i] << "Pa " << iter+1 << endl;
        }
        
        /* 
           Record the change in mass and elements
        */
        double migr_bef = vmigr*n_mass, migr_aft = vmigr_aft*n_mass;
        dm_ev[i] = migr_aft - migr_bef;
        
        tensor1d<double> el_bef = massm.transpose()*vmigr;
        tensor1d<double> el_aft = massm.transpose()*vmigr_aft;
        tensor1d<double> del = el_aft - el_bef;
        del_ev.setrowtensor(i, del);

        /* output results */
        tensor1d<double> vgm   = vgas    + vmigr;
        tensor1d<double> vgm_a = vgas_aft+ vmigr_aft;
        /*
        cout << "B: " << i << "\t";
        for (int l=0; l<(int)vmigr.size(); l++){  cout << vgas[l] + vmigr[l] << " "; }
        cout << endl;
        cout << "A: " << i << "\t";
        for (int l=0; l<(int)vmigr.size(); l++){  cout << vgas_aft[l] + vmigr_aft[l] << " "; }
        cout << endl;
        */
        
        /*
          --- Step 2:
          How dust amount

         */
        if (dm_ev[i] <= 0){
            /* 
               When migrator is evaporating, 
               --->  new n_migr = old n_migr - evaporated
            */
            ngas.setrowtensor (i, vgas_aft);
            nmigr.setrowtensor(i, vmigr_aft);
        }else{
            /* 
               When dust is condensing from gas,
               newly condensed dust will be added to both `dust` AND `migrator` phases
             */
            tensor1d<double> vdust = ndust.rowtensor(i);
            double Md = vdust*n_mass, Mp = migr_bef;
            
            double cs2 = kB*Tmd[i]/(mu*mH);
            double rd = Q0/(alpha0*cs2),    rp = St_pebble;
            
            double Sp, Sd;
            if (p_part == 0){ Sp = 0.0;    Sd = 1.0; }
            else{ Sp = Mp*pow(rd,p_part);  Sd = Md*pow(rp, p_part); }
            
            double m_migr_new = migr_bef + dm_ev[i]*(Sp/(Sp+Sd));
            double m_dust_new = Md       + dm_ev[i]*(Sd/(Sp+Sd));
            
            tensor1d<double> vmigr_new = vmigr_aft * (m_migr_new/migr_aft);
            tensor1d<double> vdust_new = vdust + vmigr_aft * (1.0 - m_migr_new/migr_aft);
            
            ngas.setrowtensor (i, vgas_aft);
            ndust.setrowtensor(i, vdust_new);
            nmigr.setrowtensor(i, vmigr_new);
            
            /* output results 
            tensor1d<double> vgdm   = vgas    + vdust     + vmigr;
            tensor1d<double> vgdm_a = vgas_aft+ vdust_new + vmigr_new; */
            
        }
    }
}
void radial::migrator_drift(double dt){
    double dr = dist[1] - dist[0];
    double s0 = 1e1;
    double rdrag = maxv(0.0,minv(1.0, abs(drift_speed(0, s0)*dt/dr)));
    
    /* if rdrag is a unity, that means either timestep or grid spacing is too large,
       leading to an unreasonable result. */
    if (rdrag == 1.0){
        cout << "large dt for drift! dr = " << dr << " , " << drift_speed(0, s0)*dt << endl; 
    }
    
    /*
      calc advection (drift) velocity first
      ... we assume that migrator (pebble) has size of St ~ 1
     */
    tensor1d<double> vd(0.0, nmigr.nrows());
    for (int i=0; i<(int)vd.size(); i++){
        double cs  = sqrt(kB*Tmd[i]/(mu*mH));  /* sound  velocity */
        double rhom = 3300, rho = sig[i]/(sqrt(2.0*pi)*cs/Omeg[i]);
        double sp = St_pebble/Omeg[i] * rho/rhom *cs;
        // cout << i << "\t " << sp << " m" << endl;
        
        vd[i] = drift_speed(i, sp);
        // cout << i << "\t" << vd[i]*dt/dr << endl;
        if (vd[i]*dt/dr >  1){ vd[i] =     dr/dt;   cout << i << "\t vd warning" << endl; }
        if (vd[i]*dt/dr < -1){ vd[i] =-1 * dr/dt;   cout << i << "\t vd warning" << endl; }
    }
    
    /*
      calc advection (drift) velocity first
     */
    Matrix<double> nmigr_update(0.0, nmigr.nrows(), nmigr.mcols());
    for (int i=0; i<(int)nmigr.nrows(); i++){
        tensor1d<double> n_update(0.0, nmigr.mcols());
        /* advection from inner region */
        if (i>0){
            if (vd[i]>0){ n_update += nmigr.rowtensor(i-1)*((vd[i]*dt/dr)*dist[i-1]); }
        }
        
        /* advection out from current grid */
        if (     vd[i]<0 and vd[i+1]<0){ n_update += nmigr.rowtensor(i)*((1.0 +          vd[i] *dt/dr)*dist[i]); }
        else if (vd[i]>0 and vd[i+1]>0){ n_update += nmigr.rowtensor(i)*((1.0 -  vd[i+1]       *dt/dr)*dist[i]); }
        else if (vd[i]<0 and vd[i+1]>0){ n_update += nmigr.rowtensor(i)*((1.0 - (vd[i+1]-vd[i])*dt/dr)*dist[i]); }
        else                           { n_update += nmigr.rowtensor(i)                               *dist[i]; }
        
        /* advection from outer region */
        if (i<(nmigr.nrows()-1)){
            if (vd[i+1]<0){ n_update += nmigr.rowtensor(i+1)*(abs(vd[i+1]*dt/dr)*dist[i+1]); }
        }
        
        /* for last grid */
        if (i==(nmigr.nrows()-1)){
            if (vd[i]<0){
                n_update += nmigr.rowtensor(i)*(square(sig[i]/sig[i-1]) * abs(vd[i]*dt/dr) * dist[i]);
            }
        }
        
        /*
          replace migartor composition with updated Matrix
          ... but, we need to adjust the radial effect b/c our model uses cylindrical coordinate system.  */
        n_update /= dist[i];
        nmigr_update.setrowtensor(i, n_update);
        
        /* check result */
        tensor1d<double> dn    = n_update - nmigr.rowtensor(i);
        tensor1d<double> dnout(0.0, nmigr.mcols()), dnin(0.0, nmigr.mcols());
        //if (vd[i]<0){ dnout += nmigr.rowtensor(i)*vd[i]*dt/dr; }
        //else{        dnin += nmigr.rowtensor(minv(i+1,(int)sig.size()-1))*vd[i-1]*dt/dr * dist[i-1]/dist[i]; }
        
        //if (vd[i+1]<0){ dnin += nmigr.rowtensor(minv(i+1,(int)sig.size()-1))*vd[i+1]*dt/dr * dist[i+1]/dist[i]; }
        //else {       dnout += nmigr.rowtensor(i)*vd[i+1]*dt/dr; }
        
        if (vd[i]>0){   dnin  += nmigr.rowtensor(maxv(0,i-1))*vd[i]*dt/dr * dist[i-1]/dist[i]; }
        if (vd[i+1]>0){ dnout += nmigr.rowtensor(i)*vd[i+1]*dt/dr; }
        
        dmigr[i] = dn*n_mass;
        dmout[i] = dnout*n_mass;
        dmin[i]  = dnin*n_mass;
        Steta[i] = vd[i]/(dist[i]*Omeg[i]);
        dPdr[i]  = (Pmd[minv(i+1,(int)sig.size()-1)]-Pmd[i])/(dist[1]-dist[0]);
    }
    vdrift = vd;
    
    
    tensor1d<double> tot0(0.0, ndust.mcols()), tot1(0.0, ndust.mcols());
    for (int i=0; i<(int)sig.size(); i++){
        for (int j=0; j<ndust.mcols(); j++){
            tot0[j] += dist[i]/AU*nmigr[i][j];
            tot1[j] += dist[i]/AU*nmigr_update[i][j];
        }
    }
    cout << "migr" << endl;
    for (int j=0; j<ndust.mcols(); j++){ cout << j << "\t" << tot0[j] << " \t -> " << tot1[j] << endl; }
    
    
    nmigr = nmigr_update;
    
}
/*
  calculate drift speed of migrator at grid i=i. 
 */
double radial::drift_speed(int i, double s){
    /*
      calculate drift speed of grain size s=s at grid i=i
     */
    
    /* 1. calculate pressure gradient at grid i=i */
    double dPdr, T_stg, sig_stg;
    if(i==0){
        dPdr = (Pmd[i] - 0.0     ) / (dist[i+1] - dist[i]);
        T_stg   = (Tmd[i] + 0.0)*0.5;
        sig_stg = (sig[i] + 0.0)*0.5;
    }else{
        dPdr    = (Pmd[i] - Pmd[i-1]) / (dist[i] - dist[i-1]);
        T_stg   = (Tmd[i] + Tmd[i-1])*0.5;
        sig_stg = (sig[i] + sig[i-1])*0.5;
    }
    
    double cs  = sqrt(kB*T_stg/(mu*mH));    /* sound  velocity */
    double vK  = dist_stg[i]*Omeg_stg[i];    /* Kepler velocity */
    
    /* 2. calculate Stokes number = (stopping time)/(eddy turnover) */
    double rhom = 3300, rho = sig_stg/(sqrt(2.0*pi)*cs/Omeg_stg[i]);
    double ts   = (rhom/rho)*(s/cs);
    double St   = ts*Omeg_stg[i];
    // cout << s << "\t" << St << "\t at " << dist[i]/AU << endl;
    
    /* 3. calculate eta = (pressure gradient)/(Gravity) */
    double eta  = (dPdr/rho)/(dist_stg[i]*Omeg_stg[i]*Omeg_stg[i]);
    //cout << i << "\t" << St/(1+St*St)*eta*vK << "\t" << eta << "\t" << dPdr << " \t" << sig[i] << endl;
    
    return St/(1+St*St)*eta*vK;
}
void radial::dust_growth(int gmdl, double dt, double tacc){
    for (int i=0; i<ndust.nrows(); i++){
        tensor1d<double> vdust = ndust.rowtensor(i);
        tensor1d<double> vmigr = nmigr.rowtensor(i);
        
        /* dust growth
           ... divide time step by `div` to describe dust-growth more accurately */
        double c_dust_rem = 1.0;
        int div = 15;
        for (int j=0; j<div; j++){
            c_dust_rem *= (1.0 - minv(1.0, dt/div/(tacc*Omeg[0]/Omeg[i])));
        }
        tensor1d<double> vdust_new = vdust*c_dust_rem;
        tensor1d<double> vmigr_new = vmigr + vdust*(1.-c_dust_rem);
                
        /* isotope */
        tensor1d<double> eldust = massm.transpose()*vdust;
        tensor1d<double> elmigr = massm.transpose()*vmigr;
        eldust *= (1-c_dust_rem);
        for (int j=0; j<(int)eldust.size(); j++){
            if ((eldust[j]+elmigr[j])>0){
                iso_migr[i][j] = (eldust[j]*iso_gasd[i][j] + elmigr[j]*iso_migr[i][j])/(eldust[j]+elmigr[j]);
            }
        }
        
        if (i == 0){
            if (c_dust_rem == 0.0){ cout << "dt too large for d->m" << c_dust_rem << endl; }
        }

        ndust.setrowtensor(i, vdust_new);
        nmigr.setrowtensor(i, vmigr_new);
    }
}
/*--------------------------------------------------------------------------------------
  clean-up the matrix
*/
void radial::remove_unresolvable(){
    for (int i=0; i<(int)ngas.nrows(); i++){
        for (int j=0; j<(int)ngas.mcols(); j++){
            if (ngas[i][j]  < DBL_MIN*10){ ngas[i][j]  = 0.0; }
            if (ndust[i][j] < DBL_MIN*10){ ndust[i][j] = 0.0; }
            if (nmigr[i][j] < DBL_MIN*10){ nmigr[i][j] = 0.0; }
            // if (nplts[i][j] < DBL_MIN*10){ nplts[i][j] = 0.0; }
        }
    }
}

/*--------------------------------------------------------------------------------------
  result output
*/
void radial::record(string filename, double time){
    filename += (to_string((int)time) + ".txt");
    ofstream fout(filename, ios::trunc);
    
    for (int i=0; i<(int)sig.size(); i++){
        /* distance + gas density */
        fout << dist[i]/AU << "\t" << sig[i] << "\t";
        
        tensor1d<double> vgas  = ngas.rowtensor(i);
        tensor1d<double> vdust = ndust.rowtensor(i);
        tensor1d<double> vmigr = nmigr.rowtensor(i);
        tensor1d<double> vplts = nplts.rowtensor(i);
        tensor1d<double> vback = vgas+vdust;
        
        /* dust/gas ratio */
        double dg_ratio = (vdust*n_mass)/(vgas*n_mass);
        fout << vmigr*n_mass << "\t" << vplts*n_mass << "\t" << no_plts[i] << "\t" << Rplts[i] << "\t";
        fout << RaccB[i]     << "\t" << RaccH[i]     << "\t";
        fout << dm_ev[i]     << "\t" << Tmd[i]       << "\t" << Pmd[i]     << "\t" << dg_ratio << "\t";
        
        for (int j=0; j<ndust.mcols(); j++){
            fout << ngas[i][j] + ndust[i][j] << "\t";
        }

        /* elemental abundance */
        tensor1d<double> elgasd = massm.transpose()*vback;
        tensor1d<double> elmigr = massm.transpose()*vmigr;
        tensor1d<double> elplts = massm.transpose()*vplts;
        for (int k=0; k<(int)elgasd.size(); k++){ fout << elgasd[k] << "\t";	}
        for (int k=0; k<(int)elmigr.size(); k++){ fout << elmigr[k] << "\t";  }
        for (int k=0; k<(int)elplts.size(); k++){ fout << elplts[k] << "\t";  }
        for (int k=0; k<(int)elmigr.size(); k++){ fout << del_ev[i][k] << "\t";	}

        /* isotope */
        for (int k=0; k<iso_gasd.mcols(); k++){ fout << iso_gasd[i][k] << "\t";	}
        for (int k=0; k<iso_migr.mcols(); k++){ fout << iso_migr[i][k] << "\t";	}
        for (int k=0; k<iso_plts.mcols(); k++){ fout << iso_plts[i][k] << "\t";	}

        fout << vmigr*n_mass << "\t" << dmigr[i] << "\t" << dmout[i] << "\t" << dmin[i] << "\t";
        fout << vdrift[i]    << "\t" << dPdr[i]  << "\t" << Steta[i] << "\t" << dm_ev[i] << "\t";

        for (int j=0; j<ndust.mcols(); j++){ fout << nmigr[i][j] << "\t"; }
        
        fout << endl;
    }
    fout.close();
}
void radial::track(string filename, double time, int loc){
    ofstream ftime(filename+"_time"+to_string(loc)+".txt", ios::app);
    
    int i=loc;
    /* distance + gas density */
    ftime << setw(12) << time << "\t" << sig[i] << "\t";
    
    tensor1d<double> vgas  = ngas.rowtensor(i);
    tensor1d<double> vdust = ndust.rowtensor(i);
    tensor1d<double> vmigr = nmigr.rowtensor(i);
    tensor1d<double> vplts = nplts.rowtensor(i);
    double dg_ratio = (vdust*n_mass)/(vgas*n_mass);
    ftime << vmigr*n_mass << "\t" << vplts*n_mass << "\t" << no_plts[i] << "\t";
    ftime << Rplts[i]     << "\t" << RaccB[i]     << "\t" << RaccH[i]   << "\t";
    ftime << dm_ev[i]     << "\t" << Tmd[i]       << "\t" << Pmd[i]     << "\t"  << dg_ratio << "\t";
    
    tensor1d<double> elplts = massm.transpose()*vplts;
    for (int k=0; k<(int)elplts.size(); k++){ ftime << elplts[k] << "\t";	}
    
    ftime << dmigr[i] << "\t" << dmout[i] << "\t" << dmin[i];
    
    ftime << endl;
    ftime.close();
    
    // cout << "distance: " << dist[loc]/AU << " at " << time << " yr" << endl;
    // cout << endl;
}
