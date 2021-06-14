/*
  pebble.cpp (part of class `radial`)
  ... functions for pebble accretion

  includes radial::streaming_instability()
  .                turbulent_concentration();
  .                pebble_accretion();
 */

#include "grid.h"

/*--------------------------------------------------------------------------------------
  pebble accretion

  streaming_instability()
  ... create planetesimal when pebble/gas ratio increases above 0.01
*/
void radial::streaming_instability(double alpha0){
    double dr = dist[1] - dist[0];
    tensor1d<double> vzero(0.0, ndust.mcols());
    
    for (int i=0; i<(int)Tmd.size(); i++){
        tensor1d<double> vgas  = ngas.rowtensor(i);
        tensor1d<double> vmigr = nmigr.rowtensor(i);
        double pgratio = (vmigr*n_mass)/(vgas*n_mass);
        double r_conc  = sqrt(St_pebble/alpha0);     /* scale height ratio of pebble and dust */
        
        if (pgratio*r_conc > 1){
            /* streaming instability operates.
               ... all pebble mass is going to be transported to planetesimal */
            
            double rd = dist[i];
            double Sarea = pi*((rd+dr)*(rd+dr) - rd*rd);
            double Mplts = Sarea*sig[i]*pgratio;
            
            double unit_mass = 4.0/3*pi*(Rplts[i]*Rplts[i]*Rplts[i])*rhom;
            no_plts[i] = Mplts/unit_mass;
            
            cout << i << "\t" << Mplts/5.972e24 << " ME \t";
            cout << no_plts[i]  <<  " of plts \t ... peb/gas = " << pgratio << endl;
            
            nmigr.setrowtensor(i, vzero);
            nplts.setrowtensor(i, vmigr);
            
            /* isotope */
            tensor1d<double> iso = iso_migr.rowtensor(i);
            iso_plts.setrowtensor(i, iso);
        }
    }
}
void radial::turbulent_concentration(double er_turb, double dyr){
    double dr = dist[1] - dist[0];
    tensor1d<double> vzero(0.0, ndust.mcols());
    
    for (int i=0; i<(int)Tmd.size(); i++){
        tensor1d<double> vgas  = ngas.rowtensor(i);
        tensor1d<double> vmigr = nmigr.rowtensor(i);
        tensor1d<double> vplts = nplts.rowtensor(i);
        double pgratio = (vmigr*n_mass)/(vgas*n_mass);
        
        double p_rand = (double)rand()/RAND_MAX;
        // cout << i << "\t" << pgratio*er_turb*dyr << " \t rand: " << p_rand << endl;
        double p_turb = er_turb;
        // if (dist[i]/AU < 1){ p_turb = er_turb*10; }
        
        if (pgratio*p_turb*dyr > p_rand){
            /* turbulent concentration triggers gravitational instability.
               ... all pebble mass is going to be transported to planetesimal */
            
            double rd = dist[i];
            double Sarea = pi*((rd+dr)*(rd+dr) - rd*rd);
            double Mpebl = Sarea*sig[i]*pgratio;
            
            double unit_mass = 4.0/3*pi*(Rplts[i]*Rplts[i]*Rplts[i])*rhom;
            if (Mpebl/unit_mass > 1){
                no_plts[i] += 1;
                
                cout << i << "\t" << Mpebl/ME << " ME \t";
                cout << no_plts[i]  <<  "of plts \t ... planetesimal mass" << endl;
                
                double           Mpebl_new = Mpebl - unit_mass;
                tensor1d<double> vmigr_new = vmigr*(Mpebl_new/Mpebl);
                tensor1d<double> vplts_new = vplts + vmigr*(unit_mass/Mpebl);
                
                nmigr.setrowtensor(i, vmigr_new);
                nplts.setrowtensor(i, vplts_new);
                
                /* isotope */
                tensor1d<double> elmigr = massm.transpose()*vmigr;
                tensor1d<double> elplts = massm.transpose()*vplts;
                elmigr *= (unit_mass/Mpebl);
                
                for (int j=0; j<(int)elmigr.size(); j++){
                    iso_plts[i][j] = (elmigr[j]*iso_migr[i][j] + elplts[j]*iso_plts[i][j])/(elmigr[j]+elplts[j]);
                }
            }
        }
    }
}
/*
  pebble_accretion()
  ... pebbles accrete onto planetesimal
  .   when planetesimal has already been created (rplts > 0) 
*/
void radial::pebble_accretion(double dt, double alpha0){
    
    tensor1d<double> vzero(0.0, ndust.mcols());
    Pmd[0] /= 100;
    for (int i=0; i<(int)Tmd.size(); i++){
        // no_plts[i] = 1;
        if (no_plts[i] > 0){
            /* planetesimal mass, ring area...  */
            double Mplts = 4.0/3*pi*(Rplts[i]*Rplts[i]*Rplts[i])*rhom;
            double rd = dist[i], dr = dist[1]-dist[0];
            double Sarea = pi*((rd+dr)*(rd+dr) - rd*rd);

            /* 1. calculate approach speed
               ... v_approach = (vpebble-vgas) + racc*Omeg */
            double dPdr;
            if (i<(nmigr.nrows()-1)){ dPdr = (Pmd[i+1] - Pmd[i])   / (dist[i+1] - dist[i]);   }
            else{                     dPdr = (Pmd[i]   - Pmd[i-1]) / (dist[i]   - dist[i-1]); }
            
            double cs  = sqrt(kB*Tmd[i]/(mu*mH));  /* sound  velocity */
            double rho = sig[i]/(sqrt(2.0*pi)*cs/Omeg[i]);
            double dvK = abs(dPdr/(2.0*rho*Omeg[i]));
            
            /* 
               --------------------------------------------------------------------------------
               
               dust accretion 
               
               2. calculate accretion radius
               ... solve using eq.(29) of Johansen and Lambrechts (2017).
               .   assume all pebbles have St of 1
               .   
               .   but note that it should not exceed the grid spacing: dr */
            double St_frag = Q0/(alpha0*cs*cs);
            double ts      = St_frag/Omeg[i];
            double racc    = accretion_radius(i, ts, dvK, Rplts[i]);
            racc = min(racc, dist[1]-dist[0]);
            RaccB[i] = racc;
            
            /* 3. calculate dust mass */
            tensor1d<double> vgas  =  ngas.rowtensor(i);
            tensor1d<double> vdust = ndust.rowtensor(i);
            double m_gas = vgas*n_mass, m_dust = vdust*n_mass;
            double dgratio = m_dust/m_gas;
            double m_dust_ring = m_dust * Sarea;
            
            /* 4. calc accretion rate */
            // double dm_dust = 2*racc*(sig[i]*dgratio)*(dvK + racc*Omeg[i]) * dt;
            double rhod    = rho*dgratio*sqrt(1 + St_frag/alpha0);
            double dm_dust = pi * (racc*racc) * rhod * (dvK+racc*Omeg[i]) * dt;
            double dm_2D   = 2  *  racc * (sig[i]*dgratio) * (dvK+racc*Omeg[i]) * dt;

            cout << dist[i]/AU << "  B: " << RaccB[i] << "\t" << St_frag << "\t" << Rplts[i] << endl;
            //cout << dist[i]/AU << "  B: " << RaccB[i] << "\t" << dm_dust/ME << "\t" << sqrt(1 + St_frag/alpha0) << "\t acc/R: " << racc/Rplts[i] << "\t r:" << dgratio << endl;
            
            /* 5. multiply by no. of planetesimals to get the actual accretion rate */
            dm_dust *= no_plts[i];
            
            /* 6. store new dust and planetesimal compositions.
               + solve for isotope */
            tensor1d<double> vplts = nplts.rowtensor(i);
            tensor1d<double> eldust = massm.transpose()*vdust;
            tensor1d<double> elplts = massm.transpose()*vplts;
            
            if (dm_dust < m_dust_ring){
                /* growth of planetesimal radius.
                   ... radius change is proportional to the cube of mass increase */
                Rplts[i] *= cbrt(1.0+(dm_dust/no_plts[i])/Mplts);
                //cout << i << "\t" << dgratio << "\t" << no_plts[i] << " -> " << dm_dust/Mplts << endl;
                
                /* composition change */
                vplts += vdust*(dm_dust/m_dust_ring);
                vdust *= (1.0 - dm_dust/m_dust_ring);
                ndust.setrowtensor(i, vdust);
                nplts.setrowtensor(i, vplts);
                
                /* isotope */
                eldust *= (1.0 - dm_dust/m_dust_ring);
                
            }else{
                Rplts[i] *= cbrt(1.0+(m_dust_ring/no_plts[i])/Mplts);
                //cout << i << "\t" << dgratio << "\t" << no_plts[i] << " -/ " << m_dust_ring/Mplts << endl;
                ndust.setrowtensor(i, vzero);
                
                vplts += vdust;
                nplts.setrowtensor(i, vplts);
            }
            
            /* isotope */
            for (int j=0; j<(int)eldust.size(); j++){
                iso_plts[i][j] = (eldust[j]*iso_gasd[i][j] + elplts[j]*iso_plts[i][j])/(eldust[j]+elplts[j]);
            }
            
            /* 
               --------------------------------------------------------------------------------
               
               migrator accretion 
               ... repeat the same thing with above, but with St_pebble instead of St_frag
               
               2. calculate accretion radius
               ... solve using eq.(29) of Johansen and Lambrechts (2017).
               .   assume all pebbles have St of 1 
               .   
               .   but note that it should not exceed the grid spacing: dr */
            ts   = St_pebble/Omeg[i];
            racc = accretion_radius(i, ts, dvK, Rplts[i]);
            racc = min(racc, dist[1]-dist[0]);
            RaccH[i] = racc;
            // cout << i << "\t" << racc << " ... accretion radius" << endl;
            
            /* 3. calculate pebble mass */
            tensor1d<double> vmigr = nmigr.rowtensor(i);
            double m_migr  = vmigr*n_mass;
            double pgratio = m_migr/m_gas;
            double m_migr_ring = m_migr * Sarea;
            
            /* 4. calc accretion rate */
            double dm_migr = 2*racc*(sig[i]*pgratio)*(dvK + racc*Omeg[i]) * dt;
            // cout << "H: " << dm_migr/ME << "\t" << Mplts/ME << "\t" << racc << "\t r:" << pgratio << endl;
            
            /* 5. multiply by no. of planetesimals to get the actual accretion rate */
            dm_migr *= no_plts[i];
            
            /* 6. store new migrator and planetesimal compositions.
               + solve for isotope */
            tensor1d<double> elmigr = massm.transpose()*vmigr;
            elplts = massm.transpose()*vplts;
            
            if (dm_migr < m_migr_ring){
                /* change planetesimal size.
                   ... Mplts -> Mplts + dm_migr, so... */
                Rplts[i] *= cbrt(1.0+(dm_migr/no_plts[i])/Mplts);
                //cout << i << "\t" << pgratio << "\t" << no_plts[i] << " -> " << dm_migr/Mplts << endl;
                
                /* composition change */
                vplts += vmigr*(dm_migr/m_migr_ring);
                vmigr *= (1.0 - dm_migr/m_migr_ring);
                nmigr.setrowtensor(i, vmigr);
                nplts.setrowtensor(i, vplts);
                
                /* isotope */
                elmigr *= (1.0 - dm_migr/m_migr_ring);
                
            }else{
                Rplts[i] *= cbrt(1.0+(m_migr_ring/no_plts[i])/Mplts);
                //cout << i << "\t" << pgratio << "\t" << no_plts[i] << " -/ " << m_migr_ring/Mplts << endl;
                nmigr.setrowtensor(i, vzero);

                vplts += vmigr;
                nplts.setrowtensor(i, vplts);
            }
            
            /* isotope */
            for (int j=0; j<(int)elmigr.size(); j++){
                iso_plts[i][j] = (elmigr[j]*iso_migr[i][j] + elplts[j]*iso_plts[i][j])/(elmigr[j]+elplts[j]);
            }
        }
        if (i > 50) exit(2);
    }
}
double radial::accretion_radius(int loc, double ts, double dvK, double rplts){
    /* 
       search for accretion radius that satisfies eq.(29) of Johansen and Lambrechts (2017)
     */
    double Mplts = 4.0/3*pi*(rplts*rplts*rplts)*rhom;
    
    double r0 = 0.0, r1 = 1.0e7;
    double f0 = solve_cubic(r0, Mplts, loc, dvK, ts);
    double f1 = solve_cubic(r1, Mplts, loc, dvK, ts);
    
    /* if the solution is not within (r0, r1), expand the search region */
    while (f1 < 0){
        r1 += 1.0e7;
        f1 = solve_cubic(r1, Mplts, loc, dvK, ts);
    }
    
    double eps = 1.0;
    while (abs(r1-r0) > eps and f0*f1 < 0){
        double rA = (r0+r1)/2.0;
        double fA = solve_cubic(rA, Mplts, loc, dvK, ts);

        if (f0*fA < 0){ r1 = rA;  f1 = fA; }
        else          { r0 = rA;  f0 = fA; }
    }
    
    double racc = (r0+r1)/2.0;
    
    /* Eq.(32) of Johansen and Lambrechts (2017) */
    double rHill = dist[loc]*pow(Mplts/Ms/3.0, 1.0/3);
    double taup = G*Mplts/cube(dvK+rHill*Omeg[loc]);
    racc *= exp(-0.4*pow(ts/taup, 0.65));
    
    
    cout << endl << dist[loc]/AU << endl;
    cout << "dvK " << dvK << " , " << rHill*Omeg[loc] << endl;
    cout << "Bondi: " << G*Mplts/(dvK*dvK) << "  hill: " << rHill << " and acc: " << (r0+r1)/2.0 << endl;
    cout << "acc: " << racc << endl;
    cout << ts/taup << " taup: "<< taup << " tboth: " << ts << " or tHil: " << G*Mplts/cube(rHill*Omeg[loc]) << endl;
    cout << "*" << exp(-0.4*pow(ts/taup, 0.65)) << endl;
    cout << "Bondi St: " << G*Mplts/(dvK*dvK*dvK)*Omeg[loc] << "\t Frag: " << ts*Omeg[loc] << "\t Hill: " << 1 << endl;
    
    return racc;
}
double radial::solve_cubic(double racc, double Mplts, int loc, double dv, double ts){
    double xiB = 0.25, xiH = 0.25;
    
    double f3 = xiH * Omeg[loc];
    double f2 = xiB * dv;
    double f0 = G*Mplts*ts;

    return (f3*racc +f2)*(racc*racc) - f0;
    
}
