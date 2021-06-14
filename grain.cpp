/*
  grain.cpp (part of class `radial`)
  ... functions for grain growth

  includes radial::diffuse();
 */

#include "grid.h"

void radial::set_grain_size_distribution(double q, double sinit, double sevo, double St){
    /* 
       this function calculates the mass ratio of each grain size
       and calculate average grain size and effective opacity
       
       1. q:     grain size distribution exponent (f(r) = r^(-q)
       2. sinit: maximum grain size where condensation front exists
       3. sevo:                     without condensation front
       
       --
       also, 
       4. St of pebble is initialized here
    */
    St_pebble = St;
    
    /*
      Solve for two sets of maximum grain size.
      Input parameters:
       1. smax:  the maximum grain size 
       2. q:     grain size distribution exponent (f(r) = r^(-q)
    */
    int no_type = 1;
    s_ave.resize(      0.0, no_type);
    opacity_ave.resize(0.0, no_type);
    
    for (int iter=0; iter<no_type; iter++){
        double smax = (!iter)? sinit : sevo;
        
        int N = 100;
        tensor1d<double> r_grain(0.0, N);
        tensor1d<double> opacity(0.0, N);
        
        /* set min/max grain sizes */
        double smin = 1e-7, lsmin = log(smin);
        double              lsmax = log(smax); /* for bouncing barrier */
        double dlns = (lsmax-lsmin)/N;
        tensor1d<double>  s(0.0, N+1), amt(0.0, N+1);
        for (int i=0; i<(N+1); i++){
            /* grain size */
            double ls  = lsmin + dlns*(double)i;
            s[i] = exp(ls);
        
            /* calculate mass amount for size = s */
            amt[i] = 4.0/3*pi*(s[i]*s[i]*s[i])*pow(s[i],-1.*q+1);
        
            /* calculate opacity for size = s[i]  */
            if (s[i]<1e-6){ opacity[i] = maxv(50., 4.84e-4*1e-6*(1e-6/s[i])); }
            else          { opacity[i] = 4.84e-4/s[i]; }
        }
        
        /* convert to mass ratio */
        double tot = 0;
        for (int i=0; i<N; i++){
            r_grain[i] = (amt[i]+amt[i+1])*dlns;
            tot       += r_grain[i];
        }
        r_grain /= tot;

        /* calc average grain size and opacity */
        double ave = 0, kpa = 0;
        for (int i=0; i<N; i++){
            ave += r_grain[i]*s[i];
            kpa += r_grain[i]*opacity[i];
        }
        
        /* store averages */
        s_ave[iter]       = ave;
        opacity_ave[iter] = kpa;
        
        cout << "average grain size: " << ave << " m ... " << iter << endl;
    }
}
void radial::grain_size_evolution(double q, double sinit, double sevo, double St){
    /* */
}
void radial::dmL_dt(double q, double rhos){
    /* */
    double r = (3-q)*(2-q)*rhos;
}
