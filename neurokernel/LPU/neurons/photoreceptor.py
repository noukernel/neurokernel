from baseneuron import BaseNeuron

import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from neurokernel.LPU.utils.simpleio import *

class Photoreceptor(BaseNeuron):
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
                self.n_photons
                )

        self.sig_cas.prepared_async_call(\
                self.gpu_grid,\
                self.gpu_block,\
                )

        self.ca_dyn.prepared_async_call(\
                self.gpu_grid,\
                self.gpu_block,\
                }

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
            

    def get_rpam_kernel(self):
        cuda_src = open('./rpam.cu', 'r')
        mod = SourceModule( cuda_src, options = ["--ptxas-options=-v"])
        func = mod.get_function("rpam")
        func.prepare( [ np.int32, # neu_num
                        np.intp   # n_photons
                        ])
        return func

    def get_sig_cas_kernel(self):
        cuda_src = open('./sig_cas.cu', 'r')
        mod = SourceModule( cuda_src, options = ["--ptxas-options=-v"])
        func = mod.get_function("signal_cascade")
        func.prepare( [ np.int32, # neu_num
                        np.intp   # n_photons
                        ])
        return func

    def get_ca_dyn_kernel(self):
        cuda_src = open('./ca_dyn.cu', 'r')
        mod = SourceModule( cuda_src, options = ["--ptxas-options=-v"])
        func = mod.get_function("calcium_dynamics")
        func.prepare( [ np.int32, # neu_num
                        np.intp, # Ca2
                        np.intp, # V_m
                        np.intp, # I_in
                        np.intp  # C_star
                        ])
        return func

    def get_gpu_kernel(self):
        self.gpu_block = (128,1,1)
        self.gpu_grid = ((self.num_neurons - 1) / self.gpu_block[0] + 1, 1)
        cuda_src = open( './photoreceptor.cu','r')
        mod = SourceModule( cuda_src,\
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
