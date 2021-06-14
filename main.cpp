/*
  main.cpp
  ... modeling a radial dynamic evolution of Mg/Si ratio in planetary disk
  
  on Apr 5, 2018 by Yoshi Miyazaki

SURF-2021-Protoplanetary-Disk-Evolution
 */

#include "./grid.h"

using namespace std;

int main(){
    /* --------------------------------------------------------
       OpenMP */
    //int NUM_THREADS = omp_get_num_procs();
    //omp_set_num_threads(NUM_THREADS);
    //cout << "no procs = " << NUM_THREADS << endl;
    
    /* --------------------------------------------------------
       [ CHANGE HERE ]
       ... initial parameter set */
    
    /* SET MODEL PARAMETERS 
     ... rin  <- the inner boundary of the radial model 
         rout <- the outer boundary of the radial model */
    double  rin = 0.05,  rout = 30;
    
    /* SET THE DISK PHYSCIS PARAMETERS
     ... sigma_in <- surface density at 1 AU (Earth-Sun distance)
         alpha    <- viscosity parameter (nu = alpha*cs*H) */
    double  sigma_in  = 2.0e5;
    double  sigma_max = sigma_in*2;
    double  alpha     = 1.0e-3;
    
    /* SET DUST GROWTH PARAMETERS
     ... sigma_in <- surface density at 1 AU (Earth-Sun distance)
         alpha    <- viscosity parameter (nu = alpha*cs*H) */
    double  sdust_init   = 1e-5;
    double  sdust_max    = 1e-5;
    double  tau_d2m      = 1e2*yr2sec;
    double  St_pebble    = 0.005;
    double  Rplts_in     = 100e3;
    double  rturb_enrich = 1e-9;
    double  p_part       = 1.0; //1/4.0;
    
    /* SET DISK CHEMISTRY */
    string  spec_list = "spec_CMAS.txt";
    string  comp_list = "comp_CMAS.txt";
    
    int aint = -1.*log10(alpha);
    int tint =     log10(tau_d2m/yr2sec);
    int Sint = -1.*log10(St_pebble/5);
    int rint = -1.*log10(rturb_enrich);
    int pint =  (p_part == 0) ? 0 : (int)(1/p_part);
    string folderno = to_string(10000*aint+1000*tint+100*Sint+10*rint+pint);
    string filename = "./case"+folderno+"/radial_";
    cout << "output to: " << filename << endl;
    
    /* ----------------------------------------------------------------*/
    /* create and initialzie the radial structure of the disk */     
    radial  model(600, rin, rout);
    
    model.set_grain_size_distribution(-3, sdust_init, sdust_max, St_pebble);
    model.set_init_density(sigma_in, sigma_max, Rplts_in, spec_list, comp_list);

    model.pebble_accretion(0.01, alpha);
    exit(2);

    double time = 0.;
    model.record(filename, time);
    
    int    max_iter = 2e4;
    double dyr = 50,  dt = dyr*yr2sec;  /* T and dust comp are updated every dyr */
    double deq = dyr*2;                   /* pebble comp      is updated every deq */
    cout << "The code runs for " << time + dyr*max_iter << " years." << endl;
    
    for (int iter=0; iter<max_iter; iter++){
        
	time += dyr;
	cout << "*" << flush;
	
	/* 
	   advect_diffuse
	   ... solve for advection + diffusion
	   .   use small timestep to maintain numerical stability
	*/
	clock_t t0 = clock();
	model.advect_diffuse(dt, alpha);
            
	/*
	  dust_growth
	  ... 1. from dust   to migrator (= pebble)
	  .   2. from pebble to planetesimal
	*/
	// model.dust_growth(0, dt, tau_d2m);
                        
	/*
	  conisder equlibrium between gas/dust and migrators
	  ... Because the opacity of migrator is very small, the mass of migrator is NOT 
	  .   considered when calcing the thermal structure. 
	*/
	//  model.migrator_drift(dt/div1);
        
	/* record a snapshot of the model */
	// model.record(filename,time);
        
        
        /* 
           record a snapshot of the model.
           also, track them chemical composition at the given location
        */
        if ((time<=1e4 and (int)time%100 == 0) or (time>1e4 and int(time)%1000==0)){  model.record(filename,time);  }
        
    }
    
    return 0;
}
