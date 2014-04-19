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
    int *rand, 
    float *p, 
    float *q 
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

    factorial = 1;
    lambda_m = n_photon[nid]/30000.0;
    for(int ii = 0; ii < km; ++ii){
        if (ii > 0){
            factorial = factorial * ii;
        }
        p[ii] = exp(-lambda_m) * (pow(lambda_m, ii)) / factorial;
    }

    int num_abs[11];
    for(int ii = 1; ii < 11; ++ii){
        num_abs[ii] = p[ii]*n_micro;
    }
    num_abs[0] = 0;
    for(int ii = 1; ii < 11; ++ii){
        for(int jj = 0; jj < num_abs[ii];++jj){
            Np[rand[jj+num_abs[ii -1]]] = ii;
        }
    }

}
""", options = ["--ptxas-options=-v"])

rpam = mod.get_function("rpam")
a = numpy.zeros(1,dtype=numpy.int32)#1000
a[0] = 1000
r_n = numpy.array(range(30000))
numpy.random.shuffle(r_n)
#r_n = numpy.zeros(1,dtype=numpy.float32)
#r_n[0] = random.random()
grid = (1,1)
dest = numpy.zeros(30000, dtype=numpy.int32)
p = numpy.zeros(20, dtype=numpy.float32)
q = numpy.zeros(20, dtype=numpy.float32)

rpam(drv.InOut(dest), drv.In(a), drv.In(r_n), drv.InOut(p), drv.InOut(q) , block=(1,1,1), grid=grid)

print a
print r_n
print "P data: ", p
print "Q data: ", q

print "dest result: ", dest
tot = 0
n_0 = 0
n_1 = 0
n_2 = 0
n_3 = 0
n_4 = 0
for ii in dest:
        tot += ii
        if(ii == 0):
                n_0 += 1
        if(ii == 1):
                n_1 += 1
        if(ii == 2):
                n_2 += 1
        if(ii == 3):
                n_3 += 1
        if(ii == 4):
                n_4 += 1

print "total number of photons absorbed: ",tot
print '1 absorbed: ', n_1
print '2 absorbed: ', n_2
print '3 absorbed: ', n_3
print '5 absorbed: ', n_4

