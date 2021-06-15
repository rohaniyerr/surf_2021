/*
  main.cpp
  ... modeling a radial dynamic evolution of Mg/Si ratio in planetary disk
  
  on Jun 14, 2021 by Yoshi Miyazaki

  SURF-2021
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
    
    string filename = "./case/radial_";
    cout << "output to: " << filename << endl;
    
    /* ----------------------------------------------------------------*/
    /* create and initialzie the radial structure of the disk */     
    radial  model(600, rin, rout);
    model.set_init_density(sigma_in, sigma_max, 0., "foo", "foo");

    model.calc_thermal_struc(alpha);
    
    double time = 0.;
    model.record(filename, time);
    
    int    max_iter = 10;
    double dyr = 1,  dt = dyr*yr2sec;  /* sigma (surface density) is updated every dyr */
    
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
           record a snapshot of the model.
           also, track them chemical composition at the given location
        */
        if ((time<=1e3) or (time>1e4 and int(time)%1000==0)){  model.record(filename,time);  }
        
    }
    
    return 0;
}
