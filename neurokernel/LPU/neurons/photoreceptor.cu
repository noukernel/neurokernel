#define NNEU &(nneu)d
#define n_micro 30000

// Get random number in previous kernel call. Generate 30000
// random numbers which are used in this code.

// Number of photons generated outside of this code. That is
// where we can do bound checks, otherwise assume fine.
__global__ void rpam(
    int neu_num,
    %(type)s *n_photon,
    %(type)s *rand,
    %(type)s *Np
    )
{
    bool not_converged = true;
    %(type)s lambda_m, n_m, fe, fa, n_m_temp;
    n_m = n_micro;
    lambda_m = 0;
    %(type)s fx[6];
    
    // FIXME:
    // Is this entire section unnecessary if we can generate
    // a poisson distribution with curand?
    while (not_converged){
        lambda_m = n_photon[nid]/n_m;
        int factorial = 1;

        // 5 max number of photons to absorb?
        for(int ii = 0; ii < 6; ++ii){
            if (ii > 0){
                factorial = factorial * ii;
            }
            fx[ii] = exp(-lambda_m) * ( pow(lambda_m,ii) / factorial );
        }
        n_m_temp = n_micro * (1 - exp(-lambda_m));
        if(fabsf(n_m_temp - n_m) < 1){
            not_converged = false;
        }

        n_m = n_m_temp;
    }
    float lambda_p = n_photon[nid] / n_m;
    int km = 10 * roundf(lambda_p+1);

    %(type)s p[km];
    %(type)s q[km];
    factorial = 1;
    float tot_p = 0.0;
    for(int ii = 0; ii < km; ++ii){
        p[ii] = exp(-lambda_p) * (pow(lambda_p, ii - 1)) / factorial;
        tot_p += p[ii];
        if (ii > 0){
            factorial = factorial * ii;
        }
    }

    float sum_p = 0.0;
    q[0] = p[0];
    for(int ii = 1; ii < km; ++ii){
        for(int jj = 1; jj < ii; ++jj){
            sum_p += p[ii];
        }
        q[ii] = sum_p / (tot_p - p[0]);
    }

    int r;
    bool found;
    int counter;
    %(type)s n_p[n_micro];
    for(int ii = 0; ii < n_m; ++ii){
        int r = rand[nid];// cuRAND somehow
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
    Np[nid] = n_p;
}

// Do something creative instead of rebuilding 7x12 matrix?
/*
#define V_00 -1
#define V_11 -1
#define V_14 1
#define V_21 1
#define V_22 -1
#define V_23 -1
#define V_32 1
#define V_36 -1
#define V_45 1
#define V_47 -1
#define V_48 -2
#define V_510 1
#define V_511 -1
#define V_68 1
#define V_69 -1
__global__ void signal_cascade(
        int neu_num,
        %(type)s *X,
        %(type)s t,
        %(type)s *T_ph,
        %(type)s *N_ph,
        %(type)s *h,
        %(type)s *c
        )
{
*/

}
