/* 
   grid.h
   
   Apr 5, 2018 by Yoshi Miyazaki
 */

#include "./GibbsFE_minimization/CGgibbsmin.h"
#include "./tensor.h"
#include <ctime>
#include <omp.h>

class radial{
 public:
    radial(int, double, double);
    
    /* model set-up */
    void set_grain_size_distribution(double, double, double, double);
    void set_init_density(double, double, double, string, string);
    
    void calc_thermal_struc(double);
    
    /* model evolution */
    void advect_diffuse(double, double);
    
    /* model output */
    void record(string, double);
    
 private:
    /* disk physical properties */
    tensor1d<double>  dist;
    tensor1d<double>  Omeg;
    tensor1d<double>  sig;
    tensor1d<double>  Tmd;
    tensor1d<double>  Pmd;
    
    /* staggered grid */
    tensor1d<double>  dist_stg;
    tensor1d<double>  Omeg_stg;
    
    /* 
       system compositions
       ... 
       ngas:  gas composition 
       ndust: solid composition showing exponential distribution (f(r) = r^-3.5) 
       nmigr: solid composition with
       nplts: planetesimal
    */
    Matrix<double>    ngas;
    Matrix<double>    ndust;
    Matrix<double>    nmigr;
    
    Matrix<double>    nplts;
    tensor1d<double>  no_plts;
    tensor1d<double>  Rplts;
    tensor1d<double>  RaccB;
    tensor1d<double>  RaccH;
    
    tensor1d<double>  dmigr;
    tensor1d<double>  dmout;
    tensor1d<double>  dmin;
    tensor1d<double>  vdrift;
    tensor1d<double>  Steta;
    tensor1d<double>  dPdr;
    
    Matrix<double>    iso_gasd;
    Matrix<double>    iso_migr;
    Matrix<double>    iso_plts;
    
    Matrix<double>    del_ev;
    tensor1d<double>  dm_ev;
    
    /* info of species */
    Matrix<double>    massm;
    tensor1d<double>  n_mass;
    string            spec_list;
    
    /* grain size distribution 
       ... evo as evolved distribution */
    tensor1d<int>     s_max;
    tensor1d<double>  s_ave;
    tensor1d<double>  opacity_ave;
    
    /* functions used inside */
    tuple<tensor1d<double>,tensor1d<double>, double> solve_middiskT(tensor1d<double>,double,double,double,int);
    tuple<tensor1d<double>,tensor1d<double>, double> solve_vertical(tensor1d<double>,double,double,double,double);
    tuple<tensor1d<double>,tensor1d<double>, double> delta_Tcube   (tensor1d<double>,double,double,double,int);
    double calc_opacity(double, double, int);

    void   grain_size_evolution(double, double, double, double);
    void   dmL_dt(double, double);
        
    double drift_speed(int, double);
    double accretion_radius(int, double, double, double);
    double solve_cubic(double, double, int, double, double);
    
    void  remove_unresolvable();

    /* differential equation + matrix solver */
    double nds(tensor1d<double>&, int);
    void   solve_implicit      (tensor1d<double>&, tensor1d<double>&, tensor1d<double>&, tensor1d<double>&);
    void   solve_Crank_Nicolson(tensor1d<double>&, tensor1d<double>&, tensor1d<double>&, tensor1d<double>&);
    void   solve_tridiag       (tensor1d<double>&, tensor1d<double>&, tensor1d<double>&, tensor1d<double>&);
    
    /* parameters used in `pebble.cpp`
       ... assume that planetesimals always have size of 100km */
    double St_pebble;
    double C0   = 1e-14;
    double Q0   = 1e0;
    double rhom = 3300;
};
