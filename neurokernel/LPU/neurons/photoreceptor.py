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
#define E0 -77
#define E1 50
#define E2 -54.387
#define g0 36
#define g1 120
#define g2 0.3

__global__ void hodgkin_huxley(
    int neu_num,
    %(type)s dt,
    %(type)s *V,
    %(type)s *I,
    %(type)s *X0,
    %(type)s *X1,
    %(type)s *X2)
{
    int bid = blockIdx.x;
    int nid = bid * NNEU + threadIdx.x;

    %(type)s v, i_ext, x0, x1, x2;

    if( nid < neu_num ){
        v = V[nid];
        i_ext = I[nid];
        x0 = X0[nid];
        x1 = X1[nid];
        x2 = X2[nid];

        v = v + ( ( i_ext - (g1*pow(x1,3)*x2*(v-50)) - (g0*pow(x0, 4)*(v+77)) - (g2*(v+54.387))) * dt);

        %(type)s a;
        a = exp(-(v+55)/10) - 1;
        if (a == 0){
            x0 = x0+((((1 - x0) * 0.1) - (x0 * 0.125 * exp(-(v+65)/80))) * dt);
        } else {
            x0 = x0+(( (1-x0) * (-0.01*(v+55)/a) - (x0 * (0.125 * exp(-(v+65)/80))) )*dt);
        }

        a = exp(-(v+40)/10)-1;
        if (a == 0){
            x1 = x1 + (( (1-x1) - (x1*(4*exp(-(v+65)/18)))) * dt);
        } else {
            x1 = x1 + (( ((1-x1) * (-0.1*(v+40)/a)) - (x1 * (4 * exp(-(v+65)/18))) ) *dt);
        }

        x2 = x2 + (( ((1 - x2)* 0.07*exp(-(v+65)/20)) - (x2 / (exp(-(v+35)/10) + 1)) ) * dt);

        V[nid] = v;
        X0[nid] = x0;
        X1[nid] = x1;
        X2[nid] = x2;
    }
    return;
}

#define n_micro = 30000.0

__global__ void rpam( 
    int neu_num,
    %(type)s *Np, 
    %(type)s *n_photon, 
    %(type)s *rand) 
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
#define uVillusVolume 3.0*powf(10.0, -12.0)
#define NaCaConst 3.0*powf(10.0, -8.0)
#define membrCap 62.8
#define ns 1 //assuming a dim background (would be 2 for bright)
#define PLCT 100
#define GT 50
#define TT 25
#define CT 0.5

__global__ void signal_cascade(
    int neu_num,
    %(type)s *I_in,
    %(type)s *V_m,
    %(type)s *Np,
    %(type)s *rand1,
    %(type)s *rand2,
    %(type)s *Ca2,
    %(type)s *X_1,
    %(type)s *X_2,
    %(type)s *X_3,
    %(type)s *X_4,
    %(type)s *X_5,
    %(type)s *X_6,
    %(type)s *X_7,
    )
{

    for(int mid = 0; mid < 30000; ++mid){

    //16: state vector:
    double X1 = X_1[neu_num];
    double X2 = X_2[neu_num];
    double X3 = X_3[neu_num];
    double X4 = X_4[neu_num];
    double X5 = X_5[neu_num];
    double X6 = X_6[neu_num];
    double X7 = X_7[neu_num];
    //double X1 = Np[mid];
    //double X2 = G;
    //double X3 = activG;
    //double X4 = activPLC;
    //double X5 = activD;
    //double X6 = activC;
    //double X7 = activT;

    double Iin = Tcurrent*X7;

    double h[12];
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
    h[10] = (CT - X6)*Ca2[neu_num];
    h[11] = X6;

    //31
    double fp = (powf((Ca2[neu_num]/Kp), mp)) / (1+powf((Ca2[neu_num]/Kp), mp));

    //32
    double fn = ns * powf((X6/Kn), mn)/(1+(powf((X6/Kn), mn)));

    double c[12];

    //19
    c[0] = DrateM * (1+hM*fn);

    //20
    c[1] = ArateG;

    //21
    c[2] = AratePLC;

    //22
    c[3] = DrateGAP;

    //23
    c[4] = DrateG;

    //24
    c[5] = ArateD;

    //25
    c[6] = DratePLC * (1+hPLC*fn);

    //26
    c[7] = DrateD*(1+hD*fn);

    //27
    c[8] = (ArateT*(1+hTpos*fp))/(ArateK*ArateK);

    //28
    c[9] = DrateT*(1+hTneg*fn);

    //29
    c[10] = CaUptakeRate/(uVillusVolume*uVillusVolume);

    //30
    c[11] = CaReleaseRate;

    //need an a vector:
    double a0 = c[0]*h[0];
    double a1 = c[1]*h[1];
    double a2 = c[2]*h[2];
    double a3 = c[3]*h[3];
    double a4 = c[4]*h[4];
    double a5 = c[5]*h[5];
    double a6 = c[6]*h[6];
    double a7 = c[7]*h[7];
    double a8 = c[8]*h[8];
    double a9 = c[9]*h[9];
    double a10 = c[10]*h[10];
    double a11 = c[11]*h[11];

    double as = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11;

    double av[12];
    av[0] = h[0]*c[0];
    double hc[12];
    hc[0] = av[0];
    int mu = 0;
    // 12 possible reaction
    for(int ii = 1; ii < 12; ++ii){
        hc[ii] = c[ii]*h[ii];
        av[ii] = av[ii-1] + hc[ii];

        if((rand2[mid]*as > av[ii - 1]) && (rand2[mid]*as <= av[ii])){
            mu = ii;
        }
    }

    if(mu == 0) {
        X_1[neu_num] += -hc[mu];
    } else if (mu == 1){
        X_2[neu_num] += -hc[mu];
        X_3[neu_num] += hc[mu];
    } else if (mu == 2){
        X_3[neu_num] += -hc[mu];
        X_4[neu_num] += hc[mu];
    } else if (mu == 3){
        X_3[neu_num] += -hc[mu];
    } else if (mu == 4){
        X_2[neu_num] += hc[mu];
    } else if (mu == 5){
        X_5[neu_num] += hc[mu];
    } else if (mu == 6){
        X_4[neu_num] += -hc[mu];
    } else if (mu == 7){
        X_5[neu_num] += -hc[mu];
    } else if (mu == 8){
        X_5[neu_num] += -2 * hc[mu];
        X_7[neu_num] += hc[mu];
    } else if (mu == 9){
        X_7[neu_num] += -hc[mu];
    } else if (mu == 10){
        X_6[neu_num] += hc[mu];
    } else {
        X_6[neu_num] += -hc[mu];
    }

    I_in[mid] = Tcurrent*X_7[neu_num];

    }
}

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
#define v 3.0*powf(10.0, -12.0)
#define n 4
#define K_r 5.5
#define K_u 30
#define C_T 0.5
#define K_Ca 100

__global__ void calciumDynamics(
    int neu_num,
	%(type)s *Ca2,
	%(type)s *V_m,
	%(type)s *I_in,
	%(type)s *C_star)
{

    double I_Ca = I_in[neu_num] * P_Ca;
    double I_NaCa = K_NaCa * (powf(Na_i,3.0) * Ca_o - powf(Na_o,3.0) * Ca2[neu_num] * exp((-V_m[neu_num]*F) / (R*T)));

    //36
    double I_CaNet = I_Ca - 2*I_NaCa;

    //41
    double f1 = K_NaCa * powf(Na_i, 3.0)*Ca_o / (v*F);
    //42
    double f2 = (K_NaCa * exp((-V_m[neu_num]*F)/(R*T)) * powf(Na_o,3.0))/(v*F);

    //40 (composed of 37,38,39)
    Ca2[neu_num] = v*(I_CaNet/(2*v*F) + n*K_r*C_star[neu_num] - f1)/(n*K_u*(C_T - C_star[neu_num]) + K_Ca - f2);

}

"""

class Photoreceptor(BaseNeuron):
    def __init__(self, n_dict, V, dt, debug=False, LPU_id=None):
        self.num_neurons = len(n_dict['id'])
        self.dt = np.double(dt)
        self.steps = max(int(round(dt / 1e-5)), 1)
        self.debug = debug
        self.LPU_id = LPU_id
        self.ddt = dt / self.steps

        # Gpot neuron Inputs/Outputs
        self.V = V
        self.X0 = garray.to_gpu( np.asarray( n_dict['X0'], dtype=np.float64 ))
        self.X1 = garray.to_gpu( np.asarray( n_dict['X1'], dtype=np.float64 ))
        self.X2 = garray.to_gpu( np.asarray( n_dict['X2'], dtype=np.float64 ))

        # FIXME: Make clean, actually use input file or something.
        # RPAM Inputs/Outputs
        self.n_photons = garray.to_gpu( np.asarray( random.randint(100,999), dtype=np.float64 ))
        self.Np = garray.to_gpu( np.zeros(30000, dtype=np.float64 ))
        r_n = np.array(range(30000))
        np.random.shuffle( r_n )
        self.rand_index = garray.to_gpu( r_n )

        # Signal Cascade Inputs/Outputs
        # FIXME: Should I_in be the same as I?
        self.rand1 = garray.to_gpu( np.random.uniform(low = 0.0, high = 1.0, size = 30000 ))
        self.rand2 = garray.to_gpu( np.random.uniform(low = 0.0, high = 1.0, size = 30000 ))
        self.Ca2 = garray.to_gpu( np.asarray( 0.00016, dtype=np.float64 ))
        # FIXME: Supposed to be Np[some id], but that doesn't exist yet...
        self.X_1 = garray.to_gpu( np.asarray( 0, dtype=np.float64 ))
        self.X_2 = garray.to_gpu( np.asarray( 50, dtype=np.float64 ))
        self.X_3 = garray.to_gpu( np.asarray( 0, dtype=np.float64 ))
        self.X_4 = garray.to_gpu( np.asarray( 0, dtype=np.float64 ))
        self.X_5 = garray.to_gpu( np.asarray( 0, dtype=np.float64 ))
        self.X_6 = garray.to_gpu( np.asarray( 0, dtype=np.float64 ))
        self.X_7 = garray.to_gpu( np.asarray( 0, dtype=np.float64 ))

        # No unique inputs/outputs for Calcium Dynamics

        # Copies an initial V into V
        cuda.memcpy_htod(int(self.V), np.asarray(n_dict['Vinit'], dtype=np.double))
        self.rpam = self.get_rpam_kernel()
        self.sig_cas = self.get_sig_cas_kernel()
        self.ca_dyn = self.get_ca_dyn_kernel()
        self.update = self.get_gpu_kernel()
        if self.debug:
            if self.LPU_id is None:
                self.LPU_id = "anon"
            self.I_file = tables.openFile(self.LPU_id + "_I.h5", mode="w")
            self.I_file.createEArray("/","array", \
                                     tables.Float64Atom(), (0,self.num_neurons))
            self.V_file = tables.openFile(self.LPU_id + "_V.h5", mode="w")
            self.V_file.createEArray("/","array", \
                                     tables.Float64Atom(), (0,self.num_neurons))
    @property
    def neuron_class(self): return True

    def eval( self, st = None):
        self.rpam.prepared_async_call(\
                self.gpu_grid,\
                self.gpu_block,\
                self.num_neurons,\
                self.n_photons.gpudata,\
                self.Np.gpudata,\
                self.rand_index.gpudata)

        self.sig_cas.prepared_async_call(\
                self.gpu_grid,\
                self.gpu_block,\
                self.num_neurons,\
                self.I.gpudata,\
                self.V,\
                self.Np.gpudata,\
                self.rand1.gpudata,\
                self.rand2.gpudata,\
                self.Ca2.gpudata,\
                self.X_1.gpudata,\
                self.X_2.gpudata,\
                self.X_3.gpudata,\
                self.X_4.gpudata,\
                self.X_5.gpudata,\
                self.X_6.gpudata,\
                self.X_7.gpudata)

        self.ca_dyn.prepared_async_call(\
                self.gpu_grid,\
                self.gpu_block,\
                self.num_neurons,\
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
            self.I.gpudata,\
            self.X0.gpudata,\
            self.X1.gpudata,\
            self.X2.gpudata)
        if self.debug:
            self.I_file.root.array.append(self.I.get().reshape((1,-1)))
            self.V_file.root.array.append(self.V.get().reshape((1,-1)))
            

    def get_rpam_kernel(self):
        #cuda_src = open('./rpam.cu', 'r')
        #mod = SourceModule( cuda_src, options = ["--ptxas-options=-v"])
        mod = SourceModule( \
                cuda_src % {"type": dtype_to_ctype(np.float64),\
                "nneu": self.gpu_block[0] },\
                options=["--ptxas-options=-v"])
        func = mod.get_function("rpam")
        func.prepare( [ np.int32, # neu_num
                        np.intp,   # n_photons
                        np.intp,   # Np
                        np.intp   # rand_index
                        ])
        return func

    def get_sig_cas_kernel(self):
        #cuda_src = open('./sig_cas.cu', 'r')
        #mod = SourceModule( cuda_src, options = ["--ptxas-options=-v"])
        mod = SourceModule( \
                cuda_src % {"type": dtype_to_ctype(np.float64),\
                "nneu": self.gpu_block[0] },\
                options=["--ptxas-options=-v"])
        func = mod.get_function("signal_cascade")
        func.prepare( [ np.int32, # neu_num
                        np.intp,   # I_in
                        np.intp,   # V_m
                        np.intp,   # Np
                        np.intp,   # rand1
                        np.intp,   # rand2
                        np.intp,   # Ca2
                        np.intp,   # X[1]
                        np.intp,   # X[2]
                        np.intp,   # X[3]
                        np.intp,   # X[4]
                        np.intp,   # X[5]
                        np.intp,   # X[6]
                        np.intp   # X[7]
                        ])
        return func

    def get_ca_dyn_kernel(self):
        #cuda_src = open('./ca_dyn.cu', 'r')
        #mod = SourceModule( cuda_src, options = ["--ptxas-options=-v"])
        mod = SourceModule( \
                cuda_src % {"type": dtype_to_ctype(np.float64),\
                "nneu": self.gpu_block[0] },\
                options=["--ptxas-options=-v"])
        func = mod.get_function("calcium_dynamics")
        func.prepare( [ np.int32, # neu_num
                        np.intp, # Ca2
                        np.intp, # V_m
                        np.intp, # I_in
                        np.intp  # C_star/X[6]
                        ])
        return func

    def get_gpu_kernel(self):
        self.gpu_block = (128,1,1)
        self.gpu_grid = ((self.num_neurons - 1) / self.gpu_block[0] + 1, 1)
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
                        np.intp, # I array
                        np.intp, # X0 array
                        np.intp, # X1 array
                        np.intp ]) # X2 array

        return func
        
    def post_run(self):
        if self.debug:
            self.I_file.close()
            self.V_file.close()
