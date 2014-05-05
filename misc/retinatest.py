import numpy as np
import networkx as nx
import h5py
from neurokernel.core import Manager
from neurokernel.LPU.LPU import LPU
from neurokernel.LPU.LPU_retina import LPU_retina
from neurokernel.tools.comm import get_random_port
from neurokernel.core import Module
from neurokernel.core import Connectivity

class MyModule(Module):
        def __init__(self, N_gpot, N_spike, port_data, port_ctrl, id=None, device=None):
                super(MyModule, self).__init__(port_data, port_ctrl, id, device)
                self.gpot_data = np.zeros(N_gpot, np.double)
                self.spike_data = np.zeros(N_spike, int)
        @property
        def N_gpot(self):
                return len(self.gpot_data)
        @property
        def N_spike(self):
                return len(self.spike_data)

        def run_step(self, in_gpot_dict, in_spike_dict, out_gpot, out_spike):
                super(MyModule, self).run_step(in_gpot_dict, in_spike_dict, out_gpot, out_spike)

port_data = get_random_port()
port_ctrl = get_random_port()
m0 = MyModule(2,4,port_data, port_ctrl)
m1 = MyModule(5,10,port_data,port_ctrl)

conn = Connectivity(m0.N_gpot, m0.N_spike, m1.N_gpot, m1.N_spike, 1, m0.id, m1.id)
conn[m0.id, 'all', :, m1.id, 'all', :] = np.ones((m0.N_gpot+m0.N_spike, m1.N_gpot+m1.N_spike))


G = nx.DiGraph()

for nn in range(6*128):
    G.add_node(1)
    G.node[nn] = {
    'model': 'Photoreceptor',
    'name' : 'neuron_0' ,
    'extern' : True,
    'public' : False,
    'spiking' : False,
    'Vinit' : -0.07,
    'SA' : 0.6982,
    'SI' : 0.000066517,
    'DRA' : 0.2285,
    'DRI' : 0.00012048 }
    nx.write_gexf(G, 'simple_lpu.gexf.gz')



(n_dict, s_dict) = LPU.lpu_parser('simple_lpu.gexf.gz')

dt = 1e-4
dur = 1.0
Nt = int(dur/dt)
start = 0.3
stop = 0.6
I_max = 0.6
t = np.arange(0, dt*Nt, dt) 
I = np.zeros((Nt, 1), dtype=np.double) 
I[np.logical_and(t>start, t<stop)] = I_max 
with h5py.File('simple_input.h5', 'w') as f: 
        f.create_dataset('array', (Nt,1), dtype=np.double, data=I) 
 
port_data = get_random_port() 
port_ctrl = get_random_port() 
 
lpu = LPU_retina(dt, n_dict, s_dict, input_file='retina_inputs.h5', output_file='retina_output.h5', port_ctrl=port_ctrl, port_data=port_data, device=0, id='simple', debug=False)
 
man = Manager(port_data, port_ctrl) 
man.add_brok() 
man.add_mod(lpu) 
man.start(steps=Nt) 
man.stop()

