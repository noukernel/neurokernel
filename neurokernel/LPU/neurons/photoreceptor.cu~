#define NNEU &(nneu)d
#define n_micro 30000.0

__global__ void rpam(int *n_photon)
{
    bool not_converged = true;
    %(type)s lambda_m, n_m;
    n_m = n_micro;
    %(type)s fx[6];
    while (not_converged){
        lambda_m = n_photon[nid]/n_m;
        int factorial = 1;
        for(int ii = 0; ii < 6; ++ii){
            if (ii > 0){
                factorial = factorial * ii
            }
            fx[ii] = exp(-lambda_m) * ( (lambda_m*ii) / factorial );
        }
        %(type)s n_m_temp = n_micro * (1 - exp(-lembda_m));
        if(n_m_temp - n_m < 1){
            not_converged = false;
        }

        n_m = n_m_temp;
    }
}
