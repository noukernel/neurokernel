import numpy as np
import rpam
import signal_cascade
import random
#import h5py

la = 1.0
ns = 1.0
N_rh = 0.0
dur = 1.0 #ms
dt = 1e-4
Nt = int(dur/dt)
t_ph = []

random.seed(1)
nphotons = random.randint(100,999)
N_ph = rpam.rpam(nphotons)

for ii in range(0, N_ph.size):
    if N_ph[ii] != 0:
        t_ph.append(ii*1e-3)

T_ph = np.asarray(t_ph)

I_in = signal_cascade.Signal_Cascade(T_ph, N_ph, N_rh, ns, la)

print I_in


#with h5py.File('simple_input.h5', 'w') as f:
#    f.create_dataset('array', (Nt, 1),
#                     dtype=np.double,
#                     data=I_in)

