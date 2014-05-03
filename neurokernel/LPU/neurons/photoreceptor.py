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
    double I,
    %(type)s *SA,
    %(type)s *SI,
    %(type)s *DRA,
    %(type)s *DRI)
{
    int bid = blockIdx.x;
    int nid = bid;

    %(type)s v, i_ext, sa, si, dra, dri;

    if( nid < neu_num ){
        v = V[nid];
        i_ext = I;
        sa = SA[nid];
        si = SI[nid];
        dra = DRA[nid];
        dri = DRI[nid];

        %(type)s inf, tau;
        // Calculate d_sa
        inf = powf(1 / (1 + exp( (-30-v) / 13.5)) , 1/3);
        tau = 0.13 + 3.39 * exp( -powf((-73-v),2) / powf(20,2) );
        SA[nid] = sa + ((inf - sa) / tau) * dt;

        // Calculate d_si
        inf = 1 / (1 + exp((-55 - v) / -5.5) );
        tau = 113 * exp( - powf((-71-v) , 2) / powf(29,2));
        SI[nid] = si + ((inf - si) / tau) * dt;

        // Calculate d_dra
        inf = powf(1 / (1 + exp((-5-v)/9) ), 0.5);
        tau = 0.5 + 5.75 * exp(-powf(-25-v, 2) / powf(32,2));
        DRA[nid] = dra + ((inf - dra) / tau) * dt;

        // Calculate d_dri
        inf = 1 / (1 + exp((-25 - v) / -10.5));
        tau = 890;
        DRI[nid] = dri + ((inf - dri) / tau) * dt;

        V[nid] = v + (((i_ext - G_K*(v - E_K) - G_Cl*(v - E_Cl) - G_s*sa*si*(v - E_K) - G_dr*dra*dri*(v - E_K) - 0.093*(v - 10)) / C) * dt);
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
#define n_micro 30000.0

#define Kp 0.3
#define Kn 0.18
#define mp 2
#define mn 3
#define m 2
#define hM 40
#define hPLC 11.1
#define hD 37.8
#define hTpos 11.5
#define hTneg 10
#define ArateG 7.05
#define AratePLC 15.6
#define ArateT 150
#define ArateD 1300
#define ArateK 100
#define DrateM 3.7
#define DrateG 3.5
#define DratePLC 144
#define DrateD 4
#define DrateT 25
#define DrateGAP 3
#define percCa 0.40
#define Tcurrent 0.68
#define NaConcExt 120
#define NaConcInt 8
#define CaConcExt 1.5
#define CaConcInt 160
#define CaMConcInt 0.5
#define nBindingSites 4
#define FaradayConst 96485
#define AbsTemp 293
#define gasConst 8.314
#define CaUptakeRate 30
#define CaReleaseRate 5.5
#define CaDiffusionRate 1000
#define NaCaConst 3.0*powf(10.0, -8.0)
#define membrCap 62.8
#define ns 1 //assuming a dim background (would be 2 for bright)
#define PLCT 100
#define GT 50
#define TT 25
#define CTconc 0.5
#define CTnum 903
#define la 0.5

#define MAX_RUN 20

__device__ void gen_rand_num(curandStateXORWOW_t *state, double* output)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	output[0] = curand_uniform(&state[tid]);
	output[1] = curand_uniform(&state[tid+1]); 
}

__device__ void gen_poisson_num(curandStateXORWOW_t *state, int* output, double lambda)
{
    int tid = (blockIdx.x * NNEU) + threadIdx.x;
	output[0] = curand_uniform(&state[tid]);
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
    int tid = bid * NNEU + threadIdx.x;
    %(type)s t_run = 0;
    int pois_num[1];
    int Np;
    int max_run;
    double h[12];
    double c[12];
    double a[12];
    double as;
	double output[2];
    int mu;
    double av[12];

    for(int nid = tid; nid < n_micro; nid += 512){
        if (nid < n_micro) {

            gen_poisson_num(state, &pois_num[0], n_photon[0]/n_micro);
            Np = pois_num[0];

           //16: state vector:
            double X1 = Np;
            double X2 = X_2[nid];
            double X3 = X_3[nid];
            double X4 = X_4[nid];
            double X5 = X_5[nid];
            double X6 = X_6[nid];
            double X7 = X_7[nid];

            max_run = 0;
            t_run = 0;
            while ((t_run < dt) && (max_run < MAX_RUN)) {
                max_run += 1;
                I_in[nid] = Tcurrent*X7;

                //18: reactant pairs - not concentrations??
                h[0] = X1;
                h[1] = X1*X2;
                h[2] = X3*(PLCT - X4);
                h[3] = X3*X4;
                h[4] = GT-X2-X3-X4;
                h[5] = X4;
                h[6] = X4; //NOT A TYPO
                h[7] = X5;
                h[8] = (X5*(X5-1)*(TT-X7))/2;
                h[9] = X7;
                h[10] = (CTnum - X6)*Ca2[nid];
                h[11] = X6;

                //31
                double fp = (powf((Ca2[nid]/Kp), mp)) / (1+powf((Ca2[nid]/Kp), mp));

                double C_star_conc = (X6/(uVillusVolume*NA))*powf(10.0,12.0);

                //32
                double fn = ns * powf((C_star_conc/Kn), mn)/(1+(powf((C_star_conc/Kn), mn)));

                c[0] = DrateM * (1 + (hM*fn) );
                c[1] = ArateG;
                c[2] = AratePLC;
                c[3] = DrateGAP;
                c[4] = DrateG;
                c[5] = ArateD;
                c[6] = DratePLC * (1 + (hPLC*fn) );
                c[7] = DrateD*(1 + (hD*fn) );
                c[8] = (ArateT*(1 + (hTpos*fp) ))/(ArateK*ArateK);
                c[9] = DrateT*(1 + (hTneg*fn) );
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
                    X_1[nid] += -1;
                } else if (mu == 1){
                    X_2[nid] += -1;
                    X_3[nid] += 1;
                } else if (mu == 2){
                    X_3[nid] += -1;
                    X_4[nid] += 1;
                } else if (mu == 3){
                    X_3[nid] += -1;
                } else if (mu == 4){
                    X_2[nid] += 1;
                } else if (mu == 5){
                    X_5[nid] += 1;
                } else if (mu == 6){
                    X_4[nid] += -1;
                } else if (mu == 7){
                    X_5[nid] += -1;
                } else if (mu == 8){
                    X_5[nid] += -2;
                    X_7[nid] += 1;
                } else if (mu == 9){
                    X_7[nid] += -1;
                } else if (mu == 10){
                    X_6[nid] += 1;
                } else {
                    X_6[nid] += -1;
                }
            }

            if(X_1[nid] < 0){
                X_1[nid] = 0;
            }
            if(X_2[nid] < 0){
                X_2[nid] = 0;
            }
            if(X_3[nid] < 0){
                X_3[nid] = 0;
            }
            if(X_4[nid] < 0){
                X_4[nid] = 0;
            }
            if(X_5[nid] < 0){
                X_5[nid] = 0;
            }
            if(X_6[nid] < 0){
                X_6[nid] = 0;
            }
            if(X_7[nid] < 0){
                X_7[nid] = 0;
            }

            I_in[nid] = Tcurrent*X_7[nid];
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
#define CaM_conc 0.5
#define n_micro 30000.0

__global__ void calcium_dynamics(
    int neu_num,
    %(type)s *I_in,
	%(type)s *Ca2,
	%(type)s *V_m,
	%(type)s *I,
	%(type)s *C_star)
{
    int bid = blockIdx.x;
    int tid = bid * NNEU + threadIdx.x;

    double I_Ca;
    double I_NaCa;
    double I_CaNet;
    double f1;
    double f2;

    for(int nid = tid; nid < n_micro; nid += 512){
        if (nid < n_micro) {

            I_Ca = I_in[nid] * P_Ca;
            I_NaCa = K_NaCa * ( (powf(Na_i,3.0) * Ca_o) - (powf(Na_o,3.0) * Ca2[nid] * exp((-V_m[neu_num]*F) / (R*T))) );

            //36
            I_CaNet = I_Ca + 2*I_NaCa;

            //41
            f1 = K_NaCa * powf(Na_i, 3.0)*Ca_o / (v*F);
            //42
            f2 = (K_NaCa * exp((-V_m[neu_num]*F)/(R*T)) * powf(Na_o,3.0))/(v*F);

            //40 (composed of 37,38,39)
            Ca2[nid] = (I_CaNet/(2*v*F) + (n*K_r* ((C_star[nid]/(v*NA))*powf(10.0,12.0)) + f1))/(n*K_u*CaM_conc + K_Ca + f2);

            //I[nid] += I_in[nid];
    
            //I[nid] = I[neu_num] * pow(10, 5);
        }
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

        # Gpot neuron Inputs/Outputs
        self.V = V
        self.SA = garray.to_gpu( np.asarray( n_dict['SA'], dtype=np.float64 ))
        self.SI = garray.to_gpu( np.asarray( n_dict['SI'], dtype=np.float64 ))
        self.DRA = garray.to_gpu( np.asarray( n_dict['DRA'], dtype=np.float64 ))
        self.DRI = garray.to_gpu( np.asarray( n_dict['DRI'], dtype=np.float64 ))

        # FIXME: Make clean, actually use input file or something.
        # RPAM Inputs/Outputs
        self.n_photons = garray.to_gpu( np.asarray( random.randint(100,999), dtype=np.float64 ))

        #r_n = np.array(range(self.num_m))
        #np.random.shuffle( r_n )
        #self.rand_index = garray.to_gpu( r_n )

        # Signal Cascade Inputs/Outputs
        # FIXME: Should I_in be the same as I?
        self.I_in = garray.to_gpu( np.zeros(self.num_m, dtype=np.float64 ))
        self.Ca2 = garray.to_gpu( np.ones( self.num_m, dtype=np.float64 )* 0.00016)

        self.X_1 = garray.to_gpu( np.zeros( self.num_m, dtype=np.float64 ))
        self.X_2 = garray.to_gpu( np.ones( self.num_m, dtype=np.float64 ) * 50)
        self.X_3 = garray.to_gpu( np.zeros( self.num_m, dtype=np.float64 ))
        self.X_4 = garray.to_gpu( np.zeros( self.num_m, dtype=np.float64 ))
        self.X_5 = garray.to_gpu( np.zeros( self.num_m, dtype=np.float64 ))
        self.X_6 = garray.to_gpu( np.zeros( self.num_m, dtype=np.float64 ))
        self.X_7 = garray.to_gpu( np.zeros( self.num_m, dtype=np.float64 ))
        self.I_HH = garray.to_gpu( np.asarray( 0.0, dtype = np.float64 ))

        # No unique inputs/outputs for Calcium Dynamics

        # Copies an initial V into V
        cuda.memcpy_htod(int(self.V), np.asarray(n_dict['Vinit'], dtype=np.double))
        self.gpu_block = (512,1,1)
        self.gpu_grid = ((self.num_neurons - 1) / self.gpu_block[0] + 1, 1)
        self.state = garray.empty(self.num_m * 2, np.float64)
        self.rand = self.get_curand_int_func()
        self.rand.prepared_async_call(self.gpu_grid, self.gpu_block, None, self.state.gpudata, 2*self.num_m, np.uint64(2))

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
                self.ddt * 1000,\
                self.state.gpudata,\
                self.I.gpudata,\
                self.I_in.gpudata,\
                self.V,\
                self.n_photons.gpudata,\
                self.Ca2.gpudata,\
                self.X_1.gpudata,\
                self.X_2.gpudata,\
                self.X_3.gpudata,\
                self.X_4.gpudata,\
                self.X_5.gpudata,\
                self.X_6.gpudata,\
                self.X_7.gpudata)

        temp_I = 0
        for ii in xrange(1, len(self.I_in)):
            temp_I += self.I_in[ii]

        self.I_HH = temp_I/15.7

        # Dirty way of debugging
        print 'X_1: ', self.X_1
        print 'X_2: ', self.X_2
        print 'X_3: ', self.X_3
        print 'X_4: ', self.X_4
        print 'X_5: ', self.X_5
        print 'X_6: ', self.X_6
        print 'X_7: ', self.X_7

        self.ca_dyn.prepared_async_call(\
                self.gpu_grid,\
                self.gpu_block,\
                st,\
                self.num_neurons,\
                self.I_in.gpudata,\
                self.Ca2.gpudata,\
                self.V,\
                self.I.gpudata,\
                self.X_6.gpudata) # X_6 is C_star

        self.update.prepared_async_call(\
            self.gpu_grid,\
            self.gpu_block,\
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
                        np.intp,   # n_photon
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
                        np.float64, # I_HH
                        np.intp, # SA array
                        np.intp, # SI array
                        np.intp, # DRA array
                        np.intp ]) # DRI array

        return func
    
    def get_curand_int_func(self):
	code = """
	#include "curand_kernel.h"
	extern "C" {
		__global__ void 
	        rand_setup(curandStateXORWOW_t* state, int size, unsigned long long seed)
	        {
	        	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	            	int total_threads = blockDim.x * gridDim.x;

	            	for(int i = tid; i < 30000; i+=size)
	            	{
        	        	curand_init(seed, i, 0, &state[i]);
    	            	}
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
