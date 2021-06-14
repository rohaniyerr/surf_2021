/*
  diffuse.cpp (part of class `radial`)
  ... functions for gas/dust diffusion

  includes radial::diffuse();
 */

#include "grid.h"

void radial::advect_diffuse(double dt, double alpha0){
    /* solve surface desnity evolution
       using implicit finite difference method. */
    int ngrid = (int)Tmd.size();
    double dr = dist[1] - dist[0];
    double div0 = 5*2,     dt0 = dt/div0;
    double div1 = 5*2,     dt1 = dt/div1;
    double div2 = 200*2,   dt2 = dt/div2;
    
    /*
      step 1. calculate viscosity, advection velocity, and element abundance
    */
    tensor1d<double> nu(0.0, ngrid), vd(0.0, ngrid), vg(0.0, ngrid), vs(0.0, ngrid);
    Matrix<double>   iso16(0.0, ngrid, iso_gasd.mcols());  /* element with 16 */
    Matrix<double>   iso18(0.0, ngrid, iso_gasd.mcols());  /* element with 18 */
    for (int i=0; i<(int)nu.size(); i++){
        double cs2 = kB*Tmd[i]/(mu*mH);
        nu[i] = alpha0*cs2/Omeg[i];
        
        /* advection occurs due to radial drift 
           ... calculate drift speed according to fragmentation barrier size */
        double St_frag = Q0/(alpha0*cs2);
        // if (alpha0 <= 4.0e-4){ St_frag *= 0.1; }
        double rho = sig[i]/(sqrt(2.0*pi)*sqrt(cs2)/Omeg[i]);
        double s0 = St_frag/Omeg[i] * rho/rhom * sqrt(cs2);
        
        /* bouncing barrier for reference
           ... it does not have a direct effect on the maximum grain size */
        // double St_boun = sqrt(sqrt(12*C0*(rhom*rhom)/(pi*alpha0*cs2*cube(sig[i]))));
        // double s0 = St_boun/Omeg[i] * rho/rhom * sqrt(cs2);
        cout << i << "\t" << s0 << "\t" << St_frag << endl;
        vd[i] = drift_speed(i, s0);
        
        /* 
           isotope
         */
        tensor1d<double> vgasd = ngas.rowtensor(i) + ndust.rowtensor(i);
        tensor1d<double> el = massm.transpose()*vgasd;
        for (int j=0; j<(int)el.size(); j++){ iso16[i][j] = el[j] * (1. - iso_gasd[i][j]); } 
        for (int j=0; j<(int)el.size(); j++){ iso18[i][j] = el[j] *       iso_gasd[i][j];  }
    }
    
    /*
      step 2.
      for surface density: discretize and create matrix
    */
    tensor1d<double> A(0.0, ngrid), B(0.0, ngrid), C(0.0, ngrid);
    
    /* A(i,i)S(i,j+1) = S(i,j)     ... A(i) */
    for (int i=0; i<ngrid; i++){
        A[i] = 6.0*dt0/(dr*dr) * nu[i] + 1.0;
    }
    /* A(i+1,i)S(i+1,j+1) = S(i,j) ... B(i) */
    for (int i=0; i<(ngrid-1); i++){
        B[i] = (-1.0/dr - 0.25/dist[i]) * (3.0*(dt0/dr)*sqrt(dist[i+1]/dist[i])*nu[i+1]);
    }
    /* A(i-1,i)S(i-1,j+1) = S(i,j) ... C(i) */
    for (int i=1; i<ngrid; i++){
        C[i] = (-1.0/dr + 0.25/dist[i]) * (3.0*(dt0/dr)*sqrt(dist[i-1]/dist[i])*nu[i-1]);
    }
    /* boundary condition: zero flux at the outer boundary */
    // C[ngrid-1] = -1.0/dr * (6.0*(dt1/dr)*sqrt(dist[ngrid-2]/dist[ngrid-1])*nu[ngrid-2]);
        
    /*tensor1d<double> tot0(0.0, ndust.mcols());
    for (int i=0; i<(int)sig.size(); i++){
        for (int j=0; j<ndust.mcols(); j++){ tot0[j] += dist[i]*(ngas[i][j]+ndust[i][j]); }
        }*/

    /* 
       step 3.
       3-1: solve for evolution
       3-2: calc vg (gas velocity)
       3-3: create sig_i evolution matrix
       3-4: solve for sig_i evolution
     */
    tensor1d<double> sig_old = sig;
    for (int k=0; k<(int)div0; k++){
        
        /* step 3-1: solve surface density evolution first */
        solve_tridiag(A,  B,  C,  sig);
        
        /* step 3-2:
           vg : factor before dc/dr : effective gas advection velocity */

        tensor1d<double> vn(0.0, ngrid);
        for (int i=0; i<ngrid; i++){
            if (i==0){
                vg[i] = 3/sqrt(dist_stg[i]) * (nu[i]*sig[i]*sqrt(dist[i]) - 0.0 );
                vg[i] /= (-0.5*dr * (0.0 + sig[i]) );
            }else{
                vg[i] = 3/sqrt(dist_stg[i]) * (nu[i]*sig[i]*sqrt(dist[i]) - nu[i-1]*sig[i-1]*sqrt(dist[i-1]));
                vg[i] /= (-0.5*dr * (sig[i-1]+sig[i]) );
            }
            vn[i] = vg[i] + vd[i];
            
            /* adjust the timestep to satisfy cdt < dr */
            double r_adv = abs(vg[i]*dt1/dr);
            while (r_adv > 1){ 
                dt1 /= 2.;  div1 *= 2.;  r_adv /= 2.;
            }
            double r_adv2 = abs(vn[i]*dt2/dr);
            while (r_adv2 > 1){ 
                dt2 /= 2.;  div2 *= 2.;  r_adv2 /= 2.;
            }
            
            // if (i>290 and i<310){ cout << i << "\t" << vg[i] << "\t" << vn[i] << "\t" << sig[i] << "\t" << Tmd[i] << "\t" << Pmd[i] << endl;}
            
        }
        
        /*
          step 3-3-1.
          for gas: discretize and create matrix
          
          solve:
          o d(Si)/dt = 1/r * d/dr[ -rvSi ] + 1/r * d/dr[ nu*r*Si* d/dr(Si/S) ]
          x dc/dt    = [ 3/sqrt(r)/sig * d/dr(nu*sig*sqrt(r)) + 1/r/sig * d/dr(nu*sig*r) ] * dc/dr  +  nu*d/dr(dc/dr)
          
        */
        tensor1d<double> Ag(0.0, ngrid), Bg(0.0, ngrid), Cg(0.0, ngrid);
        
        /* A(i,i)S(i,j+1) = S(i,j)     ... Ag(i) */
        for (int i=0; i<ngrid; i++){
            Ag[i] = -dt1/(dr*dr) * (0.5*nds(nu,i-1) + nds(nu,i) + 0.5*nds(nu,i+1))/sig[i];
            Ag[i] /= dist[i];
            if (vg[i]<0)                  { Ag[i] += vg[i]*dt1/dr; }
            if (vg[i+1]>0 and i<(ngrid-1)){ Ag[i] -= vg[i+1]*dt1/dr; }
        }
        /* A(i+1,i)S(i+1,j+1) = S(i,j) ... Bg(i) */
        for (int i=0; i<(ngrid-1); i++){
            Bg[i] =  dt1/(dr*dr) * 0.5*(nds(nu,i) + nds(nu,i+1))/sig[i+1];
            Bg[i] /= dist[i];
            if (vg[i+1]>0){ }
            else          { Bg[i] -= (dist[i+1]/dist[i]) * vg[i+1]*dt1/dr; }
        }
        /* A(i-1,i)S(i-1,j+1) = S(i,j) ... Cg(i) */
        for (int i=1; i<ngrid; i++){
            Cg[i] =  dt1/(dr*dr) * 0.5*(nds(nu,i-1) + nds(nu,i))/sig[i-1];
            Cg[i] /= dist[i];
            if (vg[i]>0){ Cg[i] += (dist[i-1]/dist[i]) * vg[i]*dt1/dr;  }
            else        { }
        }
        
        /*
          step 3-3-2.
          for dust: discretize and create matrix
        */
        tensor1d<double> Ad(0.0, ngrid), Bd(0.0, ngrid), Cd(0.0, ngrid);
                                
        /* A(i,i)S(i,j+1) = S(i,j)     ... Ag(i) */
        for (int i=0; i<ngrid; i++){
            Ad[i] = -dt2/(dr*dr) * (0.5*nds(nu,i-1) + nds(nu,i) + 0.5*nds(nu,i+1))/sig[i];
            Ad[i] /= dist[i];
            if (vn[i]<0)                  { Ad[i] += vn[i]*dt2/dr; }
            if (vn[i+1]>0 and i<(ngrid-1)){ Ad[i] -= vn[i+1]*dt2/dr; }
        }
        /* A(i+1,i)S(i+1,j+1) = S(i,j) ... Bg(i) */
        for (int i=0; i<(ngrid-1); i++){
            Bd[i] =  dt2/(dr*dr) * 0.5*(nds(nu,i) + nds(nu,i+1))/sig[i+1];
            Bd[i] /= dist[i];
            if (vn[i+1]>0){ }
            else          { Bd[i] -= (dist[i+1]/dist[i]) * vn[i+1]*dt2/dr; }
        }
        /* A(i-1,i)S(i-1,j+1) = S(i,j) ... Cg(i) */
        for (int i=1; i<ngrid; i++){
            Cd[i] =  dt2/(dr*dr) * 0.5*(nds(nu,i-1) + nds(nu,i))/sig[i-1];
            Cd[i] /= dist[i];
            if (vn[i]>0){ Cd[i] += (dist[i-1]/dist[i]) * vn[i]*dt2/dr;  }
            else        { }
        }
        
        /*
          step 3-4.
          solve for gas/dust dissipation/advection
        */
        /* solve for each species */
#pragma omp parallel for schedule(dynamic,1)
        for (int j=0; j<ndust.mcols(); j++){
            tensor1d<double> vgas  =  ngas.coltensor(j);
            tensor1d<double> vdust = ndust.coltensor(j);
            for (int k=0; k<(int)(div1/div0); k++){
                solve_Crank_Nicolson(Ag, Bg, Cg, vgas);
            }
            for (int l=0; l<(int)(div2/div0); l++){
                solve_Crank_Nicolson(Ad, Bd, Cd, vdust);
            }
                
            ngas.setcoltensor(j, vgas);
            ndust.setcoltensor(j, vdust);
        }
    }
    
    /*
      step 4.
      solve for isotopic composition
      
      el1 shows each element of `0`, where el2 `1`.
      `solve_tridiag` solves for diffusion of each element separately.
      `iso_gasd` stores the new istopic ratio after diffusion
    */
#pragma omp parallel for schedule(dynamic,1)
    for (int j=0; j<iso16.mcols(); j++){ /* j: no. of elements */
        tensor1d<double> el1 = iso16.coltensor(j);
        tensor1d<double> el2 = iso18.coltensor(j);
        
        for (int k=0; k<(int)div1; k++){ solve_tridiag(A, B, C, el1); }
        for (int k=0; k<(int)div1; k++){ solve_tridiag(A, B, C, el2); }
        
        for (int i=0; i<(int)el1.size(); i++){
            iso_gasd[i][j] = el2[i]/(el1[i]+el2[i]);
        }
    }
    
    /* 
       step 5.
       re-calculate surface density
    */
    for (int i=0; i<(int)sig.size(); i++){
        tensor1d<double> vfluid = ngas.rowtensor(i) + ndust.rowtensor(i);
        double f_mass = vfluid * n_mass;
        
        sig[i] = f_mass;
    }
    calc_thermal_struc(alpha0);
    
    /*tensor1d<double> tot1(0.0, ndust.mcols());
    for (int i=0; i<(int)sig.size(); i++){
        for (int j=0; j<ndust.mcols(); j++){ tot1[j] += dist[i]*(ngas[i][j]+ndust[i][j]); }
    }
    for (int j=0; j<ndust.mcols(); j++){ cout << j << "\t" << tot0[j] << "\t -> " << tot1[j] << endl; }*/
}

double radial::nds(tensor1d<double>& nu, int i){
    return nu[i]*dist[i]*sig[i];
}

void radial::solve_implicit(tensor1d<double>& Ao, tensor1d<double>& Bo,
                            tensor1d<double>& Co, tensor1d<double>& S){
    /* convert to implicit-solver matrix */
    int ngrid = (int)Tmd.size();
    tensor1d<double> Ai(0.0, ngrid), Bi(0.0, ngrid), Ci(0.0, ngrid);
    
    for (int i=0; i<ngrid;     i++){ Ai[i] = 1.0 - Ao[i]; }
    for (int i=0; i<(ngrid-1); i++){ Bi[i] =     - Bo[i]; }
    for (int i=1; i<ngrid;     i++){ Ci[i] =     - Co[i]; }
    
    /* solve tridiag */
    solve_tridiag(Ai, Bi, Ci, S);
}

void radial::solve_Crank_Nicolson(tensor1d<double>& Ao, tensor1d<double>& Bo,
                                  tensor1d<double>& Co, tensor1d<double>& S){
    double theta = 0.5;
    
    /* explicit side */
    int ngrid = (int)Tmd.size();
    tensor1d<double> Snew(0.0, ngrid);
    for (int i=0; i<ngrid; i++){
        Snew[i] = Co[i]*theta*S[maxv(0,i-1)] + (1+Ao[i]*theta)*S[i] + Bo[i]*theta*S[minv(i+1, ngrid-1)];
    }
    S = Snew;
    
    /* convert to implicit-solver matrix */
    tensor1d<double> Ai(0.0, ngrid), Bi(0.0, ngrid), Ci(0.0, ngrid);
    for (int i=0; i<ngrid;     i++){ Ai[i] = 1.0 - Ao[i]*(1-theta); }
    for (int i=0; i<(ngrid-1); i++){ Bi[i] =     - Bo[i]*(1-theta); }
    for (int i=1; i<ngrid;     i++){ Ci[i] =     - Co[i]*(1-theta); }
    
    /* solve tridiag */
    solve_tridiag(Ai, Bi, Ci, S);
}

void radial::solve_tridiag(tensor1d<double>& Ao, tensor1d<double>& Bo,
                           tensor1d<double>& Co, tensor1d<double>& S){
    /* result is stored in S */
    /* but for A, B,C we do not change the original vectors */
    tensor1d<double> A = Ao, B = Bo, C = Co;
    
    int imax = (int)A.size();
    if (imax !=(int)B.size() || imax !=(int)C.size()){
        cout << "diffuse.cpp/solve_tridiag: wrong vector size." << endl;
    }
    
    /* 1st row */
    B[0] /= A[0];    S[0] /= A[0];
    A[0] = 1.0;
    
    for (int j=1; j<imax; j++){
        /* swipe out C[j] to 0.0 */
        A[j] -= B[j-1] * C[j];
        S[j] -= S[j-1] * C[j];
        C[j] = 0.0;
        
        /* divide each component by A */
        B[j] /= A[j];   S[j] /= A[j];
        A[j] = 1.0;
    }
    
    /* solve diag */
    for (int j=(imax-2); j>=0; j--){  /* last row ... A[j-1] Sn[j-1] = Sb[j-1] */
        S[j] -= S[j+1] * B[j];
    }

    // for (int j=0; j<imax; j++){ if (abs(S[j]) < 1e-7){ S[j] = 0; } }
}
