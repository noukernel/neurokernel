#define NNEU &(nneu)d
#define n_micro 30000.0
// Assume n_photon random from outside the call
__global__ void rpam(
    int neu_num,
    %(type)s *n_photon
    )
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
    float lambda_p = n_photon[nid] / n_m;
    int km = 10 * roundf(lambda_p+1);

    %(type)s p[];
    %(type)s q[];
    factorial = 1;
    float tot_p = 0.0;
    for(int ii = 0; ii < km; ++ii){
        if (ii > 0){
            factorial = factorial * ii;
        }
        p[ii] = exp(-lambda_p) * (pow(lambda_p, ii - 1)) / factorial;
        tot_p += p[ii];
    }

    float sum_p = 0.0;
    for(int ii = 0; ii < km; ++ii){
        for(int jj = 0; jj < ii; ++jj){
            sum_p += p[ii];
        }
        q[ii] = sum_p / (tot_p - p[0]);
    }

    int r;
    bool found;
    int counter;
    %(type)s n_p[n_micro];
    for(int ii = 0; ii < n_m; ++ii){
        int r = // cuRAND somehow
        found = false;
        counter = 0;
        while(!found && counter < q){ //q size, whatever that is
            if (r < q_counter){
                n_p[ii] = counter - 2;
                found = true;
            }
            counter += 1;
        }
    }
}
