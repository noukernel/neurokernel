import numpy as np
import rpam
import signal_cascade
import random

la = 1
ns = 1
t = 1 #ms
dt = 10**(-4)
T_ph = np.zeros(t/dt)

random.seed(1)
nphotons = random.randint(100,999)
N_ph = rpam.rpam(nphotons)
I_in = signal_cascade.Signal_Cascade(T_ph, N_ph, N_rh, ns, la)

