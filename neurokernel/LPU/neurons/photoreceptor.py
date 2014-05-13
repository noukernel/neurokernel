from baseneuron import BaseNeuron

import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from neurokernel.LPU.utils.simpleio import *

import random

cuda_src = """
#define NNEU %(nneu)d
#define E_K -85
#define E_Cl -30
#define G_s 1.6
#define G_dr 3.5
#define G_Cl 0.056
#define G_K 0.082
#define C 4

__global__ void hodgkin_huxley(
    int neu_num,
    %(type)s dt,
    %(type)s *V,
    %(type)s *I,
    %(type)s *SA,
    %(type)s *SI,
    %(type)s *DRA,
    %(type)s *DRI)
{
    int bid = blockIdx.x;
    int nid = (bid * 128) + threadIdx.x;

    %(type)s v, i_ext, sa, si, dra, dri, ddt;

    if( nid < neu_num ){
        v = V[nid] * 1000;
        i_ext = I[nid] / 15.7;
        sa = SA[nid];
        si = SI[nid];
        dra = DRA[nid];
        dri = DRI[nid];
        ddt = dt/10;

	for(int run_dmc = 0; run_dmc < 10; ++run_dmc)
	{

        	%(type)s inf, tau;
        	// Calculate d_sa
       		inf = powf(1 / (1 + exp( (-30-v) / 13.5)) , 1.0/3.0);
        	tau = 0.13 + 3.39 * exp( -powf((-73-v),2) / powf(20,2) );
        	sa = sa + ((inf - sa) / tau) * ddt;

        	// Calculate d_si
        	inf = 1 / (1 + exp((-55 - v) / -5.5) );
        	tau = 113 * exp( - powf((-71-v) , 2) / powf(29,2));
        	si = si + ((inf - si) / tau) * ddt;

        	// Calculate d_dra
        	inf = powf(1 / (1 + exp((-5-v)/9) ), 0.5);
        	tau = 0.5 + 5.75 * exp(-powf(-25-v, 2) / powf(32,2));
        	dra = dra + ((inf - dra) / tau) * ddt;

        	// Calculate d_dri
        	inf = 1 / (1 + exp((-25 - v) / -10.5));
        	tau = 890;
        	dri = dri + ((inf - dri) / tau) * ddt;

        	v = v + (((i_ext - G_K*(v - E_K) - G_Cl*(v - E_Cl) - G_s*sa*si*(v - E_K) - G_dr*dra*dri*(v - E_K) - 0.093*(v - 10)) / C) * ddt);
	}

	SA[nid] = sa;
	SI[nid] = si;
	DRA[nid] = dra;
	DRI[nid] = dri;
	V[nid] = v / 1000.0;
    }
    return;
}

"""

sig_cas_src = """

#include "curand_kernel.h"
#define NNEU %(nneu)d

extern "C" {



#define NA 6.02*powf(10.0,23.0)
#define uVillusVolume 3.0*powf(10.0, -9.0)
#define n_microf 30000.0
#define n_micro 30000

#define Kp 0.3
#define Kn 0.18
#define mp 2.0
#define mn 3.0
#define m 2.0
#define hM 40.0
#define hPLC 11.1
#define hD 37.8
#define hTpos 11.5
#define hTneg 10.0
#define Kappa_Gstar 7.05
#define Kappa_PLCstar 15.6
#define Kappa_Tstar 150.0
#define Kappa_Dstar 1300.0
#define K_Dstar 100.0
#define Gamma_Mstar 3.7
#define Gamma_G 3.5
#define Gamma_PLCstar 144.0
#define Gamma_Dstar 4.0
#define Gamma_Tstar 25.0
#define Gamma_GAP 3.0
#define percCa 0.40
#define Tcurrent 0.68
#define NaConcExt 120.0
#define NaConcInt 8.0
#define CaConcExt 1.5
#define CaConcInt 160.0
#define CaMConcInt 0.5
#define nBindingSites 4.0
#define FaradayConst 96485.0
#define AbsTemp 293.0
#define gasConst 8.314
#define CaUptakeRate 30.0
#define CaReleaseRate 5.5
#define CaDiffusionRate 1000.0
#define NaCaConst 3.0*powf(10.0, -8.0)
#define membrCap 62.8
#define ns 2.0 //assuming a dim background (would be 2 for bright)
#define PLCT 100
#define GT 50
#define TT 25
#define CTconc 0.5
#define CTnum 903
#define la 0.5

__device__ void gen_rand_num(curandStateXORWOW_t *state, double* output)
{
    int tid = threadIdx.x + (blockIdx.x*blockDim.x);
	
	output[0] = curand_uniform(&state[tid]);
	output[1] = curand_uniform(&state[tid]); 
}

__device__ void gen_poisson_num(curandStateXORWOW_t *state, int* output, double lambda)
{
    int tid = threadIdx.x + (blockIdx.x*blockDim.x);
	output[0] = curand_poisson(&state[tid], lambda);
}

__global__ void signal_cascade(
    int neu_num,
    %(type)s dt,
    curandStateXORWOW_t* state,
    %(type)s *I,
    %(type)s *I_in,
    %(type)s *V_m,
    %(type)s *n_photon,
    %(type)s *Ca2,
    %(type)s *X_1,
    %(type)s *X_2,
    %(type)s *X_3,
    %(type)s *X_4,
    %(type)s *X_5,
    %(type)s *X_6,
    %(type)s *X_7)
{

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    %(type)s t_run = 0;
    int pois_num[1];
    int Np;
    double h[12];
    double c[12];
    double a[12];
    double as;
	double output[2];
    int mu;
    double av[12];

    for(int nid = tid; nid < n_micro; nid += 512){
        if (nid < n_micro) {

            gen_poisson_num(state, &pois_num[0], n_photon[nid + (n_micro * bid)]/n_microf);
            Np = pois_num[0];
            X_1[nid + (n_micro * bid)] += Np;

            t_run = 0;
            while (t_run < dt) {

                //18: reactant pairs - not concentrations??
                h[0] = X_1[nid + (n_micro * bid)];
                h[1] = X_1[nid + (n_micro * bid)]*X_2[nid + (n_micro * bid)];
                h[2] = X_3[nid + (n_micro * bid)]*(PLCT - X_4[nid + (n_micro * bid)]);
                h[3] = X_3[nid + (n_micro * bid)]*X_4[nid + (n_micro * bid)];
                h[4] = GT-X_2[nid + (n_micro * bid)]-X_3[nid + (n_micro * bid)]-X_4[nid + (n_micro * bid)];
                h[5] = X_4[nid + (n_micro * bid)];
                h[6] = X_4[nid + (n_micro * bid)]; //NOT A TYPO
                h[7] = X_5[nid + (n_micro * bid)];
                h[8] = (X_5[nid + (n_micro * bid)]*(X_5[nid + (n_micro * bid)]-1)*(TT-X_7[nid + (n_micro * bid)]))/2.0;
                h[9] = X_7[nid + (n_micro * bid)];
                h[10] = (CTnum - X_6[nid + (n_micro * bid)])*Ca2[nid + (n_micro * bid)];
                h[11] = X_6[nid + (n_micro * bid)];

                //31
                double fp = (powf((Ca2[nid + (n_micro * bid)]/Kp), mp)) / (1+powf((Ca2[nid + (n_micro * bid)]/Kp), mp));

                double C_star_conc = (X_6[nid + (n_micro * bid)]/(uVillusVolume*NA))*powf(10.0,12.0);

                //32
                double fn = ns * powf((C_star_conc/Kn), mn)/(1+(powf((C_star_conc/Kn), mn)));

                c[0] = Gamma_Mstar * (1 + (hM*fn) );
                c[1] = Kappa_Gstar;
                c[2] = Kappa_PLCstar;
                c[3] = Gamma_GAP;
                c[4] = Gamma_G;
                c[5] = Kappa_Dstar;
                c[6] = Gamma_PLCstar * (1 + (hPLC*fn) );
                c[7] = Gamma_Dstar*(1 + (hD*fn) );
                c[8] = (Kappa_Tstar*(1 + (hTpos*fp) ))/(K_Dstar*K_Dstar);
                c[9] = Gamma_Tstar*(1 + (hTneg*fn) );
                c[10] = CaUptakeRate;
                c[11] = CaReleaseRate;

                //need an a vector:
                as = 0;
                for(int ii = 0; ii < 12; ++ii) {
                    a[ii] = c[ii]*h[ii];
                    as += a[ii];
                }
    
    	        gen_rand_num(state, &output[0]);

	            // Calculate next dt step
                t_run += (1 / (la + as)) * logf(1/output[0]);

                av[0] = a[0];
                mu = 0;
                // 12 possible reaction
                for(int ii = 1; ii < 12; ++ii){
                    av[ii] = av[ii-1] + a[ii];
                    if((output[1]*as > av[ii - 1]) && (output[1]*as <= av[ii])){
                        mu = ii;
                    }
                }
                if(mu == 0) {
                    X_1[nid + (n_micro * bid)] += -1;
                } else if (mu == 1){
                    X_2[nid + (n_micro * bid)] += -1;
                    X_3[nid + (n_micro * bid)] += 1;
                } else if (mu == 2){
                    X_3[nid + (n_micro * bid)] += -1;
                    X_4[nid + (n_micro * bid)] += 1;
                } else if (mu == 3){
                    X_3[nid + (n_micro * bid)] += -1;
                } else if (mu == 4){
                    X_2[nid + (n_micro * bid)] += 1;
                } else if (mu == 5){
                    X_5[nid + (n_micro * bid)] += 1;
                } else if (mu == 6){
                    X_4[nid + (n_micro * bid)] += -1;
                } else if (mu == 7){
                    X_5[nid + (n_micro * bid)] += -1;
                } else if (mu == 8){
                    X_5[nid + (n_micro * bid)] += -2;
                    X_7[nid + (n_micro * bid)] += 1;
                } else if (mu == 9){
                    X_7[nid + (n_micro * bid)] += -1;
                } else if (mu == 10){
                    X_6[nid + (n_micro * bid)] += 1;
                } else {
                    X_6[nid + (n_micro * bid)] += -1;
                }
                if(X_1[nid + (n_micro * bid)] < 0){
                    X_1[nid + (n_micro * bid)] = 0;
                }
                if(X_2[nid + (n_micro * bid)] < 0){
                    X_2[nid + (n_micro * bid)] = 0;
                }
                if(X_3[nid + (n_micro * bid)] < 0){
                    X_3[nid + (n_micro * bid)] = 0;
                }
                if(X_4[nid + (n_micro * bid)] < 0){
                    X_4[nid + (n_micro * bid)] = 0;
                }
                if(X_5[nid + (n_micro * bid)] < 0){
                    X_5[nid + (n_micro * bid)] = 0;
                }
                if(X_6[nid + (n_micro * bid)] < 0){
                    X_6[nid + (n_micro * bid)] = 0;
                }
                if(X_7[nid + (n_micro * bid)] < 0){
                    X_7[nid + (n_micro * bid)] = 0;
                }
            }
            I_in[nid + (n_micro * bid)] = Tcurrent*X_7[nid + (n_micro * bid)];
	    
        }
    }
    return;
}
}
"""

ca_dyn_src = """

#define NNEU %(nneu)d

#define NA 6.02*powf(10.0,23.0)
#define Kp 0.3
#define Kn 0.18
#define mp 2
#define mn 3
#define Na_o 120
#define Na_i 8
#define Ca_o 1.5
#define Ca_id 160.0*powf(10.0, -6.0)
#define P_Ca 0.4
#define K_NaCa 3.0*powf(10.0, -8.0)
#define F 96485
#define R 8.314
#define T 293
#define v 3.0*powf(10.0, -9.0)
#define n 4
#define K_r 5.5
#define K_u 30
#define C_T 903
#define K_Ca 1000
#define C_T_conc 0.5
#define n_microf 30000.0
#define n_micro 30000

__global__ void calcium_dynamics(
    int neu_num,
    %(type)s *I_in,
	%(type)s *Ca2,
	%(type)s *V_m,
	%(type)s *I,
	%(type)s *C_star)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    double I_Ca;
    double I_NaCa;
    double I_CaNet;
    double f1;
    double f2;
    double CaM_conc;
    __shared__ double I_sum[512];
    I_sum[tid] = 0;

    for(int nid = tid; nid < n_micro; nid += 512){
        //if (nid < n_micro) {

            CaM_conc = (C_T_conc - (C_star[nid + (n_micro * bid)]/(v*NA)*powf(10.0,12.0)));

            I_Ca = I_in[nid + (n_micro * bid)] * P_Ca;
            I_NaCa = K_NaCa * ( (powf(Na_i,3.0) * Ca_o) - (powf(Na_o,3.0) * Ca2[nid + (n_micro * bid)] * exp((-V_m[bid]*F) / (R*T))) );

            //36
            I_CaNet = I_Ca + 2*I_NaCa;

            //41
            f1 = K_NaCa * powf(Na_i, 3.0)*Ca_o / (v*F);
            //42
            f2 = (K_NaCa * exp((-V_m[bid]*F)/(R*T)) * powf(Na_o,3.0))/(v*F);

            //40 (composed of 37,38,39)
            Ca2[nid + (n_micro*bid)] = (I_CaNet/(2*v*F) + (n*K_r* ((C_star[nid + (n_micro * bid)]/(v*NA))*powf(10.0,12.0)) + f1))/(n*K_u*CaM_conc + K_Ca + f2);

            if(Ca2[nid + (n_micro * bid)] < 0){
                Ca2[nid + (n_micro * bid)] = 0;
            }

	    I_sum[tid] += I_in[nid + (n_micro*bid)];

       // }
    }
    __syncthreads();

    if (tid < 32){
	    for (int ii = 1; ii < 16; ++ii){
	        I_sum[tid] += I_sum[tid + ii*32];
	    }
    }
    __syncthreads();

    if (tid == 0){
	    for (int ii = 1; ii < 32; ++ii){
    	    I_sum[tid] += I_sum[ii];
	    }
    	I[bid] = I_sum[tid];
    }
    return;
}

"""
# FIXME above: What is the actual scaling factor for LIC?


class Photoreceptor(BaseNeuron):
    def __init__(self, n_dict, V, dt, debug=False, LPU_id=None):
        self.num_neurons = len(n_dict['id'])
        self.dt = np.double(dt)
        self.steps = max(int(round(dt / 1e-5)), 1)
        self.debug = debug
        self.LPU_id = LPU_id
        self.ddt = dt / self.steps

        self.num_m = 30000 # number microvilii
        self.num_threads = 512

        # Gpot neuron Inputs/Outputs
        self.V = V
        self.H_V = np.asarray(n_dict['Vinit'], dtype = np.double)
        self.SA = garray.to_gpu( np.asarray( n_dict['SA'], dtype=np.float64 ))
        self.SI = garray.to_gpu( np.asarray( n_dict['SI'], dtype=np.float64 ))
        self.DRA = garray.to_gpu( np.asarray( n_dict['DRA'], dtype=np.float64 ))
        self.DRI = garray.to_gpu( np.asarray( n_dict['DRI'], dtype=np.float64 ))

        if 'photon_input' in n_dict:
            self.photon_input = garray.to_gpu( np.asarray( n_dict['photon_input'], dtype=np.float64 ))
        else:
            self.photon_input = garray.to_gpu( np.ones( self.num_m * self.num_neurons, dtype=np.float64 ) * 30)
        # Signal Cascade Inputs/Outputs
        # FIXME: Should I_in be the same as I?
        self.I_in = garray.to_gpu( np.zeros(self.num_m * self.num_neurons, dtype=np.float64 ))
        self.Ca2 = garray.to_gpu( np.ones( self.num_m * self.num_neurons, dtype=np.float64 )* 0.00016)

        self.X_1 = garray.to_gpu( np.zeros( self.num_m * self.num_neurons, dtype=np.float64 ))
        self.X_2 = garray.to_gpu( np.ones( self.num_m * self.num_neurons, dtype=np.float64 ) * 50)
        self.X_3 = garray.to_gpu( np.zeros( self.num_m * self.num_neurons, dtype=np.float64 ))
        self.X_4 = garray.to_gpu( np.zeros( self.num_m * self.num_neurons, dtype=np.float64 ))
        self.X_5 = garray.to_gpu( np.zeros( self.num_m * self.num_neurons, dtype=np.float64 ))
        self.X_6 = garray.to_gpu( np.zeros( self.num_m * self.num_neurons, dtype=np.float64 ))
        self.X_7 = garray.to_gpu( np.zeros( self.num_m * self.num_neurons, dtype=np.float64 ))
        self.I_HH = garray.to_gpu( np.zeros( self.num_neurons, dtype = np.float64 ))

        # No unique inputs/outputs for Calcium Dynamics

        # Copies an initial V into V
        cuda.memcpy_htod(int(self.V), np.asarray(n_dict['Vinit'], dtype=np.double))
        self.gpu_block = (self.num_threads,1,1)
        self.gpu_grid = (self.num_neurons, 1)
        self.state = garray.empty(self.num_threads * self.num_neurons * 12, np.int32)
        self.rand = self.get_curand_int_func()
        self.rand.prepared_async_call(self.gpu_grid, self.gpu_block, None, self.state.gpudata, self.num_threads, np.uint64(2))

        self.sig_cas = self.get_sig_cas_kernel()
        self.ca_dyn = self.get_ca_dyn_kernel()
        self.update = self.get_gpu_kernel()
        if self.debug:
            if self.LPU_id is None:
                self.LPU_id = "anon"
            self.I_file = tables.openFile(self.LPU_id + "_I_inside.h5", mode="w")
            self.I_file.createEArray("/","array", \
                                     tables.Float64Atom(), (0,self.num_neurons))

    @property
    def neuron_class(self): return True

    def eval( self, st = None):
        self.sig_cas.prepared_async_call(\
                self.gpu_grid,\
                self.gpu_block,\
                st,\
                self.num_neurons,\
                self.dt,\
                self.state.gpudata,\
                self.I.gpudata,\
                self.I_in.gpudata,\
                self.V,\
                self.photon_input.gpudata,\
                self.Ca2.gpudata,\
                self.X_1.gpudata,\
                self.X_2.gpudata,\
                self.X_3.gpudata,\
                self.X_4.gpudata,\
                self.X_5.gpudata,\
                self.X_6.gpudata,\
                self.X_7.gpudata)

        #self.I_HH = garray.sum(self.I_in)
        #print garray.sum(self.I_in) / 15.7


        # Dirty way of debugging
        #print 'X_1: ', self.X_1
        #print 'X_2: ', self.X_2
        #print 'X_3: ', self.X_3
        #print 'X_4: ', self.X_4
        #print 'X_5: ', self.X_5
        #print 'X_6: ', self.X_6
        #print 'X_7: ', self.X_7

        #printing V
        cuda.memcpy_dtoh(self.H_V, int(self.V))
        print self.H_V

        self.ca_dyn.prepared_async_call(\
                self.gpu_grid,\
                self.gpu_block,\
                st,\
                self.num_neurons,\
                self.I_in.gpudata,\
                self.Ca2.gpudata,\
                self.V,\
                self.I_HH.gpudata,\
                self.X_6.gpudata) # X_6 is C_star

        self.update.prepared_async_call(\
            ((self.num_neurons / 128) + 1,1),\
            (128, 1, 1),\
            st,\
            self.num_neurons,\
            self.ddt * 1000,\
            self.V,\
            self.I_HH.gpudata,\
            self.SA.gpudata,\
            self.SI.gpudata,\
            self.DRA.gpudata,\
            self.DRI.gpudata)
        if self.debug:
            self.I_file.root.array.append(self.I.get().reshape((1,-1)))
            
    def get_sig_cas_kernel(self):
        #cuda_src = open('./sig_cas.cu', 'r')
        #mod = SourceModule( cuda_src, options = ["--ptxas-options=-v"])
        mod = SourceModule( \
                sig_cas_src % {"type": dtype_to_ctype(np.float64),\
                "nneu": self.gpu_block[0] },\
                options=["--ptxas-options=-v"],no_extern_c = True)
        func = mod.get_function("signal_cascade")
        func.prepare( [ np.int32, # neu_num
                        np.float64, # dt
                        np.intp, # rand state
                        np.intp,    # I
                        np.intp,   # I_in
                        np.intp,   # V_m
                        np.intp,   # photon_input
                        np.intp,   # Ca2
                        np.intp,   # X[1]
                        np.intp,   # X[2]
                        np.intp,   # X[3]
                        np.intp,   # X[4]
                        np.intp,   # X[5]
                        np.intp,   # X[6]
                        np.intp])   # X[7]
        return func

    def get_ca_dyn_kernel(self):
        #cuda_src = open('./ca_dyn.cu', 'r')
        #mod = SourceModule( cuda_src, options = ["--ptxas-options=-v"])
        mod = SourceModule( \
                ca_dyn_src % {"type": dtype_to_ctype(np.float64),\
                "nneu": self.gpu_block[0] },\
                options=["--ptxas-options=-v"])
        func = mod.get_function("calcium_dynamics")
        func.prepare( [ np.int32, # neu_num
                        np.intp, # I_in
                        np.intp, # Ca2
                        np.intp, # V_m
                        np.intp, # I
                        np.intp])  # C_star/X[6]
        return func

    def get_gpu_kernel(self):
        #cuda_src = open( './photoreceptor.cu','r')
        mod = SourceModule( \
                cuda_src % {"type": dtype_to_ctype(np.float64),\
                "nneu": self.gpu_block[0] },\
                options=["--ptxas-options=-v"])
        #mod = SourceModule( cuda_src,\
        #        options=["--ptxas-options=-v"])
        func = mod.get_function("hodgkin_huxley")
        func.prepare( [ np.int32, # neu_num
                        np.float64, # dt
                        np.intp, # V array
                        np.intp, # I_HH
                        np.intp, # SA array
                        np.intp, # SI array
                        np.intp, # DRA array
                        np.intp ]) # DRI array

        return func
    
    def get_curand_int_func(self):
	code = """
    #include "curand_kernel.h"
    extern "C" {
        __global__ void rand_setup(curandStateXORWOW_t* state, int size, unsigned long long seed)
        {
            int tid = threadIdx.x;
            curand_init(seed, tid, 0, &state[tid]);
        }
    }
    	"""
    	mod = SourceModule(code, no_extern_c = True)
    	func = mod.get_function("rand_setup")
    	func.prepare([np.intp, np.int32, np.uint64])
    	return func

        
    def post_run(self):
        if self.debug:
            self.I_file.close()
