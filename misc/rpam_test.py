#define NNEU &(nneu)d
#define n_micro 30000

# Get random number in previous kernel call. Generate 30000
# random numbers which are used in this code.

# Number of photons generated outside of this code. That is
# where we can do bound checks, otherwise assume fine.

import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule
import random

mod = SourceModule("""
__global__ void rpam(
    int *Np,
    int *n_photon,
    float *rand
    )
{
    int nid = threadIdx.x;
    int n_micro = 30000;
    bool not_converged = true;
    float lambda_m, n_m, fe, fa, n_m_temp;
    n_m = n_micro;
    lambda_m = 0;
    float fx[6];
    
    int factorial = 1;
    // FIXME:
    // Is this entire section unnecessary if we can generate
    // a poisson distribution with curand?
    while (not_converged){
        lambda_m = n_photon[nid]/n_m;

        // 5 max number of photons to absorb?
        for(int ii = 0; ii < 6; ++ii){
            if (ii > 0){
                factorial = factorial * ii;
            }
            fx[ii] = exp(-lambda_m) * ( pow(lambda_m,ii) / factorial );
        }
        n_m_temp = n_micro * (1 - exp(-lambda_m));
        if( fabsf(n_m_temp - n_m) < 1){
            not_converged = false;
        }

        n_m = n_m_temp;
    }
    float lambda_p = n_photon[nid] / n_m;
    int km = 10 * roundf(lambda_p+1);

    float p[km];
    float q[km];
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
    q[0] = 0;
    for(int ii = 1; ii < km; ++ii){
        for(int jj = 1; jj < ii; ++jj){
            sum_p += p[ii];
        }
        q[ii] = sum_p / (tot_p - p[0]);
    }

    int r;
    bool found;
    int counter;
    int n_p[n_micro];
    for(int ii = 0; ii < n_micro; ++ii){
        int r = rand[nid];// cuRAND somehow
        found = false;
        counter = 1;
        n_p[ii] = 0;
        while(!found){
            if (r < q[counter]){
                n_p[ii] = counter - 1;
                found = true;
            }
            counter += 1;
        }
    }
    Np[nid] = n_p;
}

}
""", options = ["--ptxas-options=-v"])

rpam = mod.get_function("rpam")
a = 1000
grid = (1,1)
dest = numpy.zeros(30000)

rpam(drv.Out(dest), drv.In(a), drv.In(random.random()), block=(1,1,1), grid=grid)

print dest
