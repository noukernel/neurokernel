# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import random

def Signal_Cascade(T_ph, N_ph, N_rh, ns, la):

    # Parameter Values
    K_p = 0.3
    K_n = 0.18
    m_p = 2
    m_n = 3
    h_Mstar = 40
    h_PLCstar = 11.1
    h_Dstar = 37.8
    h_TstarP = 11.5
    h_TstarN = 10
    Kappa_Gstar = 7.05
    Kappa_PLCstar = 15.6
    Kappa_Dstar = 1300
    Kappa_Tstar = 150
    K_Dstar = 100
    Gamma_Mstar = 3.7
    Gamma_Gstar = 3.5
    Gamma_G = 3.5 #FIXME: This is not specified ANYWHERE
    Gamma_PLCstar = 144
    Gamma_Dstar = 4
    Gamma_Tstar = 25
    Gamma_GAP = 3
    T_T = 25
    G_T = 50
    PLC_t = 100
    P_Ca = 0.4
    I_Tstar = 0.68
    Ct = 0.5
    Na_con_o = 120
    Na_con_i = 8
    Ca_con_o = 1.5
    Ca_con_id = 160
    C_con_i = 0.5
    n = 4
    F = 96485
    T = 293
    R = 8.314
    K_U = 30
    K_R = 5.5
    K_Ca = 1000
    v = 3e-12
    K_NaCa = 3e-8
    C_m = 62.8
    Ca2 = 0.0 # FIXME: This probably doesn't even make sense

    Vm = -65
    f1 = K_NaCa * (Na_con_i**3)*(Ca_con_o**2)/(v*F)
    f2 = K_NaCa * np.exp(-Vm*F/R/T) * (Na_con_o**3)/(v*F)
    
    '''
    X defined as follows:
    X = [M*, G, G*, PLC*, D*, C*, T*]
    '''
    #FIXME: add non-zero initial values
    X = np.zeros(7)
    
    #FIXME: should be Ca_con_i, not Ca_con_id...but value is ambiguous
    fp = ((Ca_con_id/K_p)**m_p)/(1+(Ca_con_id/K_p)**m_p)
    fn = ns*((X[5]/K_n)**m_n)/(1+(X[5]/K_n)**m_n)
    
    V = np.zeros((7, 12));
    V[0,0] = -1;
    V[1,1] = -1;
    V[2,1] = 1;
    V[2,2] = -1;
    V[2,3] = -1;
    V[3,2] = 1;
    V[3,6] = -1;
    V[4,5] = 1;
    V[4,7] = -1;
    V[4,8] = -2;
    V[5,10] = 1;
    V[5,11] = -1;
    V[6,8] = 1;
    V[6,9] = -1;
    
    #FIXME: Find Ca2 and CaM, replace h[10]
    h = np.array([X[0], X[0]*X[1], X[2]*(PLC_t - X[3]), X[2]*X[3], (G_T - X[2] - X[1] - X[3]), X[3], X[3], X[4], 0.5*(X[4]*(X[4]-1)*(T_T-X[6])), X[6], P_Ca*K_Ca, X[5]])
    
    c = np.array([Gamma_Mstar*(1+h_Mstar*fn), Kappa_Gstar, Kappa_PLCstar, Gamma_GAP, Gamma_G, Kappa_Dstar, Gamma_PLCstar*(1+h_PLCstar*fn), Gamma_Dstar*(1+h_Dstar*fn), Kappa_Tstar*(1+h_TstarP*fp)/(Kappa_Dstar**2), Gamma_Tstar*(1+h_TstarN*fn), K_U/(v**2), K_R])
    
    ii = 1 # Iterator
    t = 0 # time
    t_end = 5 # end-time
    mu = 0 #reaction to perform each dt
    a_v = np.zeros((1,len(h)))
    
    while (t < t_end):
        random.seed(1)#FIXME: Randomize seed after testing
        r1 = random.random()
        r2 = random.random()
    
        #as is a function in python. changed to a_s
        a_s = np.dot(h,c)
        #interpreting ln as the natural log
        dt = (1/(la + a_s)) * np.log(1/r1)
        
        if ((t + dt) > T_ph(ii)):
            t = T_ph(ii)
            ii = ii + 1
            N_rh = N_rh + N_ph(t)
        else:
            t = t + dt
            
        
        for k in range(0,a_v.size):
            a_v[k] = np.sum(np.dot(h[0:k],c[0:k])) 
        
            if (r2*a_s > a_v[k] & r2*a_s < a_v[k+1]): 
                mu = k
        
        for m in range(0,X.size):
            X[mu] = X[mu] + V[mu,m]*h[m]*c[m]
        
        h = np.array([X[0], X[0]*X[1], X[2]*(PLC_t - X[3]), X[3]*X[4], (G_T - X[3] - X[2] - X[4]), X[4], X[4], X[5], .5*(X[5]*(X[5]-1)*(T_T-X[7])), X[7], Ca2*CaM, X[6]])
        
        #  Do Calcium Stuff  
        I_in = I_Tstar * X[6]
        I_Ca = P_Ca * I_in
        I_NaCa = K_NaCa*(((Na_i**3) * (Ca_o**2)) - ((Na_o**3)*Ca2*np.exp(V_m*F/R/T)))
        I_ca_net = I_Ca - 2*I_NaCa
        
        fp = ((Ca2/K_p)**m_p)/(1+(Ca2/K_p)**m_p)
        fn = ns*((X[5]/K_n)**m_n)/(1+(X[5]/K_n)**m_n)
        
        c = np.array([Gamma_Mstar*(1+h_Mstar*fn), Kappa_Gstar, Kappa_PLCstar, Gamma_GAP, Gamma_G, Kappa_Dstar, Gamma_PLCstar*(1+h_PLCstar*fn), Gamma_Dstar*(1+h_Dstar*fn), Kappa_Tstar*(1+h_TstarP*fp)/(Kappa_Dstar**2), Gamma_Tstar*(1+h_TstarN*fn), K_U/(v**2), K_R]);

    return I_in
        
Signal_Cascade(1, 2, 3, 4, 5)

# <codecell>


