#define NNEU %(nneu)d
#define n_micro = 30000.0

__global__ void rpam( 
    int neu_num,
    %(type)s *Np, 
    %(type)s *n_photon, 
    %(type)s *rand, 
    %(type)s *p
    ) 
{ 
    bool not_converged = true; 
    %(type)s lambda_m, n_m, fe, fa, n_m_temp; 
    n_m = n_micro; 
    lambda_m = 0; 
    float fx[6]; 
     
    int factorial = 1; 

    factorial = 1;
    lambda_m = n_photon[neu_num]/n_micro;
    for(int ii = 0; ii < 20; ++ii){
        if (ii > 0){
            factorial = factorial * ii;
        }
        p[ii] = exp(-lambda_m) * (pow(lambda_m, ii)) / factorial;
    }

    int num_abs[20];
    for(int ii = 1; ii < 20; ++ii){
        num_abs[ii] = p[ii]*n_micro;
    }
    num_abs[0] = 0;
    for(int ii = 1; ii < 20; ++ii){
        for(int jj = 0; jj < num_abs[ii];++jj){
            Np[rand[jj+num_abs[ii -1]]] = ii;
        }
    }

}
