/*
  vertical.cpp
  ... calculate mid-disk temperature from the surface temperature.
  
  includes 1. physical parameters (r, T, P)
           2. composition (n_dust, n_migrator, n_planetesimal)
*/

#include "grid.h"

tuple<tensor1d<double>, tensor1d<double>, double> radial::solve_vertical(tensor1d<double> vtot, double Tp,
                                                                         double Sigma, double Omega, double alpha){
    /* calcualte physical vertical structure */
    double scale = 5.0,   cs = sqrt(kB*Tp/(mu*mH));
    double H     = cs/Omega, nu = alpha*cs*H;
    double rho0  = Sigma/(sqrt(2.0*pi)*H);

    double Te = sqrt(sqrt(9.0/8*nu*Sigma*Omega*Omega/sb));
    double Tc = Te/sqrt(sqrt(2.0)), Fc = 9.0/8.0*nu*Sigma*Omega*Omega;
    tensor1d<double> vgas, vsol;
    tensor1d<double> vgas_tot(0.0, ndust.mcols()), vsol_tot(0.0, ndust.mcols());
    double r_tot;

    /* where is this grid? */
    double rAU   = cbrt(G*Ms/(Omega*Omega))/AU;
    int    igrid = round(rAU*20) - 1;
    string filename = "./vertical_" + to_string(igrid) + ".txt";
    
    /* ofstream fout(filename, ios::trunc);
       if (igrid > 20 and igrid < 50){ fout << nu << "\t" << Te << "\t" << Tp << endl; } */
    // cout << rAU << "\t" << Omega << " \t" << Te << " \t @" << igrid << endl;
    
    /* Calculate temperature from surface towards the mid-plane */
    int m = 30;  double dz = scale*H/m;
    for (int i=(m-1); i>=0; i--){
        /* pressure */
        double ri = exp(-square(double(i)/(m-1)*scale)/2.0);
        double rhoi = rho0*exp(-square(double(i)/(m-1)*scale)/2.0);
        double Pc = rhoi*kB*Tc/(mu*mH);
        
        /* condensation calculation - determine n_dust at grid(i) */
        MoleculeData_G gibbs_list(spec_list, Tc, Pc);
        gibbsminCG     min(gibbs_list, vtot);
        vgas = min.getngas_or_solid(0);
        vsol = min.getngas_or_solid(2);
        double dgratio = (vsol*n_mass)/(vgas*n_mass);
        
        /* opacity calculation */
        double kappa = calc_opacity(Tc, dgratio, 0);
        
        /* temperature calculation */
        double Tj=Tc, Fj=Fc;  /* Tj, Fj: T&flux at bottom of each section */
        int    divT = 100;   /* calculate T in a finer grid to avoid numerical error */
        for (int l=divT; l>0; l--){
            rhoi = rho0*exp(-square((double(i)+double(l)/divT)/(m-1)*scale)/2.0);
            double gamma = 3.0*kappa*rhoi / (16.0*sb*(pow(Tj,3.0)));
            Tj  += gamma*Fj*dz/divT;                      /* dT/dz */
            Fj  -= 9.0/4.0*rhoi*nu*Omega*Omega*dz/divT;   /* dF/dz */
        }
        
        double dTaccept = 100;
        /* if (Tj-Tc)<dTaccept: T difference between z_(i-1) & z_i is small enough,
           no detailed calc is necessary. Just use Tj as T@z_(i-1) */
        
        /* if (Tj-Tc)>dTaccept: divide grid & calculate w/ finer mesh to avoid any temperature jump */
        double Tpc = Tj;  /* store T calculated w/ no divided grid */
        int    div = 2;   /* divide grid into div (start w/ 2)     */
        while (div<32 && (Tj-Tc)>dTaccept){
            /* initialize Tj w/ Tc (T@ upper edge of the grid) */
            /* Tj is updated at each section and will be the T@lower edge of the grid
               Tc is stored as T@z_i at the end of calc
               !! Note that T@z_i means T@z=z_i (lower edge of grid(i)) */
            Tj = Tc; Fj = Fc;
            
            /* avoid huge T jump @each subsection */
            double T1up = Tc, dTm = 0;  /* T1up stores T@upper edge of subgrid */
            for (int j=(div-1); j>=0; j--){  /* z=dz*(i+j/div)
                                                T is calced from up to below, meaning w/ DECREASing j */
                /* set subdivided composition into the grid */
                MoleculeData_G gibbs_list(spec_list, Tc, Pc);
                gibbsminCG     min(gibbs_list, vtot);
                vgas = min.getngas_or_solid(0);
                vsol = min.getngas_or_solid(2);
                double dgratio = (vsol*n_mass)/(vgas*n_mass);

                /* opacity calculation */
                double kappa = calc_opacity(Tc, dgratio, 0);
                
                /* temperature calculation w/ fine mesh */
                for (int l=divT; l>0; l--){
                    double rhoi = rho0*exp(-square((double(i)+double(j+double(l)/divT)/div)/(m-1)*scale)/2.0);
                    double gamma = 3.0*kappa*rhoi / (16.0*sb*(pow(Tj,3.0)));
                    Tj  += gamma*Fj*dz/div/divT;                    /* dT/dz */
                    Fj  -= 9.0/4.0*rhoi*nu*Omega*Omega*dz/div/divT; /* dF/dz */
                }
		
                /* calc T difference in this subgrid. dT should be smaller than 200K */
                dTm  = maxv(Tj-T1up,dTm); T1up = Tj;
                // cout << j << "  Tj: " << Tj << "\t Fj: " << Fj << endl;
            }

            /* subgrid of div and div/2 should result in T difference below 2K
               AND dT in each subgrid should be smaller than 200K */
            if(abs(Tj-Tpc)<2 && dTm < 200){ break; }  
            else{
                /* If not, try with a finer subgrid */
                div=div*2; Tpc = Tj;
            }
        }
        
        /* store T and Fc, n_gas, n_dust & mass change [per year!] */
        Tc = Tj;  Fc = Fj;     /* Tj: Temperautre for grid BELOW */
        vgas_tot += vgas*ri;
        vsol_tot += vsol*ri;
        r_tot    += ri;

        /* output vertical structure to trackt the results */
        // if (igrid > 20 and igrid < 50){ fout << i << "\t" << Tc << "\t" << Fc << endl; }
        // cout << i << "  Tc: " << Tc << "\t Fc: " << Fc << endl;
    }
    vgas_tot /= r_tot;
    vsol_tot /= r_tot;
    
    return forward_as_tuple(vgas, vsol, Tc);
}
