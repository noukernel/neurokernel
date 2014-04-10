%% Parameter Values
K_p = 0.3;
K_n = 0.18;
m_p = 2;
m_n = 3;
h_Mstar = 40;
h_PLCstar = 11.1;
h_Dstar = 37.8;
h_TstarP = 11.5;
h_TstarN = 10;
Kappa_Gstar = 7.05;
Kappa_PLCstar = 15.6;
Kappa_Dstar = 1300;
Kappa_Tstar = 150;
K_Dstar = 100;
Gamma_Mstar = 3.7;
Gamma_G = 3.5;
Gamma_PLCstar = 144;
Gamma_Dstar = 4;
Gamma_Tstar = 25;
Gamma_GAP = 3;
T_T = 25;
G_T = 50;
PLC_T = 100;
C_T = 0.5;
P_Ca = .4;
I_Tstar = 0.68;
Na_o = 120;
Na_i = 8;
Ca_o = 1.5;
Ca_id = 160;
C_i = 0.5;
n = 4;
F = 96485;
T = 293;
R = 8.314;
K_u = 30;
K_r = 5.5;
K_Ca = 1000;
v = 3e-12;
K_NaCa = 3e-8;
C_m = 62.8;
V_m = -70e-3;
f1 = K_NaCa*((Na_i*v)^3)*(((Ca_o)*v)^2)/v/F;
f2 = K_NaCa*(exp(-V_m*F/R/T))*((Na_o*v)^3)/v/F;