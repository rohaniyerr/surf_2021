#include "./CGgibbsmin.h"

int main(){
    double T = 1413.77;
    double P = 58.6824;
    
    string spec_list = "../radial_drift/spec_CMAS.txt";
    MoleculeData_G gibbs_list(spec_list, T, P);
    int ncomp = gibbs_list.getnumofMolecule();
    tensor1d<double> vcomp(0.0, ncomp);

    /* input */
    vcomp[0] = 61.5916;  // Al
    vcomp[1] = 46.3249;  // Ca
    vcomp[2] = 10074;  // Mg
    vcomp[3] = 9099.07;  // Si
    vcomp[4] = 6267.9;  // Fe(g)
    vcomp[5] = 983.693;  // Na
    vcomp[6] = 246648000./2;  // H2
    vcomp[7] = 121810./2;  // O2
    vcomp[8] = 0.0;
    vcomp[9] = 0.0;
    vcomp[10] = 0.0;
    vcomp[11] = 0.0;
    vcomp[12] = 0.0;
    vcomp[13] = 0.0;
    vcomp[14] = 0.0;
    vcomp[15] = 0.0;
    vcomp[16] = 0.0;
    vcomp[17] = 0.0;
    vcomp[18] = 0.0;
    vcomp[19] = 0.0;
    vcomp[20] = 0.0;
    vcomp[21] = 0.0;
    
    cout << "A" << endl;
    gibbsminCG     min(gibbs_list, vcomp);
    cout << "A" << endl;
    
    string file = "out.txt";
    min.result(T, P, file);

    return 0;
}
