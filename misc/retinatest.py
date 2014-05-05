import argparse
import numpy as np
import networkx as nx
import h5py
from neurokernel.core import core
from neurokernel.base import base
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

dt = 1e-4
dur = 1.0
Nt = int(dur/dt)
start = 0.3
stop = 0.6
I_max = 0.6

parser = argparse.ArgumentParser()
parser.add_argument('--debug', default=False,
                    dest='debug', action='store_true',
                    help='Write connectivity structures and inter-LPU routed data in debug folder')
parser.add_argument('-l', '--log', default='none', type=str,
                    help='Log output to screen [file, screen, both, or none; default:none]')
parser.add_argument('-s', '--steps', default=steps, type=int,
                    help='Number of steps [default: %s]' % steps)
parser.add_argument('-d', '--port_data', default=None, type=int,
                    help='Data port [default: randomly selected]')
parser.add_argument('-c', '--port_ctrl', default=None, type=int,
                    help='Control port [default: randomly selected]')
parser.add_argument('-a', '--lam_dev', default=0, type=int,
                    help='GPU for lamina lobe [default: 0]')

args = parser.parse_args()

file_name = None
screen = False
if args.log.lower() in ['file', 'both']:
    file_name = 'neurokernel.log'
if args.log.lower() in ['screen', 'both']:
    screen = True
logger = base.setup_logger(file_name, screen)

if args.port_data is None and args.port_ctrl is None:
    port_data = get_random_port()
    port_ctrl = get_random_port()
else:
    port_data = args.port_data
    port_ctrl = args.port_ctrl

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

