# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import math
import numpy as np
import random as rnd
n_micro = 30000.0

def rpam(n_photon, seed=rnd.randint(1,1000)):
    print seed
    rnd.seed(seed)
    if(n_photon >= 1000):
        print 'input to rpam must be less than 1000, input is ', n_photon
        n_photon = 999
    elif(n_photon < 100):
        print 'intput to rpam must be greater than 100, input is ', n_photon
        n_photon = 100

    global n_micro
    n_m = n_micro # number of activated microvilli
    fe = 0 # fraction of microvilli that escape activation
    fa = 0 # fraction of microvilli that are activated
    lambda_m = 0 # average number of photons absorbed per microvillus

    x = [0,1,2,3,4,5]
    fx = {}
    converged = True
    while(converged):
        lambda_m = n_photon/n_m

        for ii in x:
            fx[ii] = math.exp(-lambda_m) * ((lambda_m**ii)/math.factorial(ii))

        fe = math.exp(-lambda_m)
        fa = 1 - fe
        n_m_temp = n_micro*fa

        if(abs(n_m_temp - n_m) < 1):
            converged = False

        n_m = n_m_temp

    lambda_p = n_photon/n_m
    km = 10*round(lambda_p + 1)
    p = np.zeros(km, dtype=np.double)
    q = np.zeros(km, dtype=np.double)

    for ii in range(0,((int(km))-1)):
        p[ii] = math.exp(-lambda_p) * (lambda_p**(ii - 1)) / math.factorial(ii)

    for ii in range(1,((int(km)) - 1)):
        q[ii] = sum(p[1:ii])/(sum(p) - p[0])

    n_p = np.zeros(int(n_micro))
    for nn in range(1,int(math.floor(n_m))):
        r = rnd.random()
        found = False
        counter = 0
        while(not found and counter < q.size):
            if r < q[counter]:
                n_p[nn] = counter - 2;
                found = True
            counter += 1
    N = np.zeros(4)
    for nn in range(0,int(math.floor(n_m))):
        for mm in range(0,3):
            if n_p[nn] == mm:
                N[mm] += 1


    return n_p
'''
nphotons = 980
N = rpam(nphotons)
print 'our result: ', N
'''
# <codecell>


