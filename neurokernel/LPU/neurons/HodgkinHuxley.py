from baseneuron import BaseNeuron

import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from neurokernel.LPU.utils.simpleio import *

# E0 -> E_Na
# E1 -> E_K
# E2 -> E_L
# g0 -> g_Na
# g1 -> g_K
# g2 -> g_L

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
"""

class HodgkinHuxley(BaseNeuron):
    def __init__(self, n_dict, V, dt, debug=False, LPU_id=None):
        self.num_neurons = len(n_dict['id'])
        self.dt = np.double(dt)
        self.steps = max(int(round(dt / 1e-5)), 1)
        self.debug = debug
        self.LPU_id = LPU_id
        self.ddt = dt / self.steps

        self.V = V
        self.X0 = garray.to_gpu( np.asarray( n_dict['X0'], dtype=np.float64 ))
        self.X1 = garray.to_gpu( np.asarray( n_dict['X1'], dtype=np.float64 ))
        self.X2 = garray.to_gpu( np.asarray( n_dict['X2'], dtype=np.float64 ))

        # Copies an initial V into V
        cuda.memcpy_htod(int(self.V), np.asarray(n_dict['Vinit'], dtype=np.double))
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
        self.update.prepared_async_call(\
            self.gpu_grid,\
            self.gpu_block,\
            st,\
            self.V,\
            self.num_neurons,\
            self.ddt * 1000,\
            self.I.gpudata,\
            self.X0.gpudata,\
            self.X1.gpudata,\
            self.X2.gpudata)
        if self.debug:
            self.I_file.root.array.append(self.I.get().reshape((1,-1)))
            self.V_file.root.array.append(self.V.get().reshape((1,-1)))
            

    def get_gpu_kernel(self):
        self.gpu_block = (128,1,1)
        self.gpu_grid = ((self.num_neurons - 1) / self.gpu_block[0] + 1, 1)
        #cuda_src = open( './hodgkin_huxley.cu','r')
        mod = SourceModule( \
                cuda_src % {"type": dtype_to_ctype(np.float64),\
                            "nneu": self.gpu_block[0] },\
                options=["--ptxas-options=-v"])
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
