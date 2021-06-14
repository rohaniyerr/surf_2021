

#include "./CGgibbsmin.h"

int main(){
    double T = 1503.44;
    double P = 39.7486;
    
    string spec_list = "spec_CMAS.txt";
    MoleculeData_G gibbs_list(spec_list, T, P);
    int ncomp = gibbs_list.getnumofMolecule();
    tensor1d<double> vcomp(0.0, ncomp);

    /* input */
    vcomp[0] = 12.7388;  // Al
    vcomp[1] = 13.9242;  // Ca
    vcomp[2] = 6866.17;  // Mg
    vcomp[3] = 6766.64;  // Si
    vcomp[4] = 5293.71;  // Fe(g)
    vcomp[5] = 825.134;  // Na
    vcomp[6] = 206871000./2;  // H2
    vcomp[7] = 98769.4/2;  // O2
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
