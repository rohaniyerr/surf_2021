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
}
void radial::set_init_density(double sigma_in, double sigma_max, double Rplts_in, string spec_list_in, string comp_list){
    
    /* set density */
    double sigma_min = 1e2;
    for (int i=0; i<(int)sig.size(); i++){
        sig[i] = sigma_in*(1.0*AU)/dist[i];
        
        if (sig[i]>sigma_max){ sig[i] = sigma_max; }
        if (dist[i]/AU > 15 ){ sig[i] = sigma_min; }
    }
        
    /* display disk mass */
    double Mdisk = 0.0, dr = dist[1]-dist[0];
    for (int i=0; i<(int)sig.size(); i++){ Mdisk += (2*pi*dist[i]*dr)*sig[i]; }
    cout << "Mdisk = " << Mdisk/Ms << endl;
    
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
	Tmd[i] = 100;
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
        fout << dist[i]/AU << "\t" << sig[i] << "\t" << Tmd[i] << endl;;
    }
    fout.close();
}
