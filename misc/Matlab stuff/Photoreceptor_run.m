close all;

%% Random Photon Absorption Model

%dt = 1;
%t = 0:dt:1; % Only doing 1 'time step'

%NT = zeros(30000,length(t));
%Nphoton = 1000;

%RPAM;

% Should we store separate I_in per microvillus? No idea...
%I_in = zeros(30000,1);
%{
T_ph = cell(30000,1);
for i = 1:30000
    T_ph{i} = cell(2,1);
    
    if NT(i,1) ~= 0
        T_ph{i}{1} = dt;
        T_ph{i}{2} = NT(i,1);
    else
        T_ph{i}{1} = 0;
        T_ph{i}{2} = 0;
    end
end
%}

%% Initialize Parameters

% la between 0.1 and 1
% ns = 1 if dim and 2 if bright

t = 0;
tend = 1;
la = 0.5;
n_s = 2;

K_p = 0.3;
K_n = 0.18;
m_p = 2;
m_n = 3;
M = 2;
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
I_Tstar = .68;
Na_o = 120;
Na_i = 8;
Ca_o = 1.5;
Ca_dark = 160e-6;
C_i = 0.5;
n = 4;
F = 96485;
T = 293;
R = 8.314;
K_u = 30;
K_r = 5.5;
K_Ca = 1000;
v = 3e-9;
K_NaCa = 3e-8;
C_m = 62.8;
V_m = -70e-3;
f1 = K_NaCa*((Na_i)^3)*((Ca_o)^2)/v/F;
f2 = K_NaCa*(exp(-V_m*F/R/T))*((Na_o)^3)/v/F;
NA = 6.02e23;
sa = 0.6982;
si = 6.6517e-5;
dra = 0.2285;
dri = 1.2048e-4;

%% Initialization of Matrices

X = cell(7,1);
Ca2 = zeros(1,1);
mu = zeros(1,1);
I_in = zeros(1,1);

%for ii = 1:30000

for m = 1:7
    X{m} = zeros(1,1);
end

X{1}(1) = Np(1);
%X{1} = zeros(1,1001);
%X{1}(1:100) = 1;
X{2}(1) = 50;
X{3}(1) = 0;
X{4}(1) = 0;
X{5}(1) = 0;
X{6}(1) = 0;
X{7}(1) = 0;

Ca2(1) = Ca_dark; 
C_star_conc = X{6}(1)/v/NA*10^12;

% Hill functions for positive and negative calcium feedback
fp = ((Ca2(1)/K_p)^m_p)/(1+(Ca2(1)/K_p)^m_p);
fn = n_s*((C_star_conc/K_n)^m_n)/(1+(C_star_conc/K_n)^m_n);

h = [X{1}(1); X{1}(1)*X{2}(1); X{3}(1)*(PLC_T - X{4}(1)); X{3}(1)*X{4}(1); 
    (G_T - X{3}(1) - X{2}(1) - X{4}(1)); X{4}(1); X{4}(1); X{5}(1); 
    .5*(X{5}(1)*(X{5}(1)-1)*(T_T-X{7}(1))); X{7}(1); Ca2(1)*(903 - X{6}(1)); X{6}(1)];

c = [Gamma_Mstar*(1+h_Mstar*fn); Kappa_Gstar; Kappa_PLCstar; Gamma_GAP; 
    Gamma_G; Kappa_Dstar; Gamma_PLCstar*(1+h_PLCstar*fn); 
    Gamma_Dstar*(1+h_Dstar*fn); Kappa_Tstar*(1+h_TstarP*fp)/Kappa_Dstar^2;
    Gamma_Tstar*(1+h_TstarN*fn); K_u; K_r];


I_in(1) = I_Tstar * X{7}(1);      % Initialize input current (X(7,1) is Tstar)
I_Ca = P_Ca * I_in(1);         % Calcium Current ~40%
I_NaCa = K_NaCa*(((Na_i^3)*(Ca_o^2)) - ((Na_o^3)*Ca2(1)*exp(-V_m*F/R/T)));
I_Canet = I_Ca + 2*I_NaCa;

i = 1;          % X iterator (how many times Signal_Cascade was called)
%ii = 1;         % T_ph iterator (next photon absorption)
%tt = 0;         % Timekeeper

%For one microvillus
while (t < tend)
    
    RPAM;
    X{1}(i) = Np(1);
    
    X_old = [X{1}(i), X{2}(i), X{3}(i), X{4}(i), X{5}(i), X{6}(i), X{7}(i)];
    
    [Z, dt] = Signal_Cascade(X_old, h, c);
    t = t + dt;
    
    %tt = tt + 1;%t_new;
    
    % No idea, but won't matter for now
    %{
    if abs == true
        ii = ii + 1;
    end
    %}
    
    %mu = [mu nu];
    
    i = i + 1;
    
    for m = 1:7
    %for m = 2:7
        X{m} = [X{m} Z(m)];%#ok<*AGROW>
    end
    
    C_star_conc = X{6}(i)/v/NA*10^12;
        
    % Update h
    h = [X{1}(i); X{1}(i)*X{2}(i); X{3}(i)*(PLC_T - X{4}(i)); X{3}(i)*X{4}(i); 
    (G_T - X{3}(i) - X{2}(i) - X{4}(i)); X{4}(i); X{4}(i); X{5}(i); 
    .5*(X{5}(i)*(X{5}(i)-1)*(T_T-X{7}(i))); X{7}(i); Ca2(i-1)*(903 - X{6}(i)); X{6}(i)];
    
    % Update fp, fn
    fp = ((Ca2(i-1)/K_p)^m_p)/(1+(Ca2(i-1)/K_p)^m_p);
    fn = n_s*((C_star_conc/K_n)^m_n)/(1+(C_star_conc/K_n)^m_n);
    
    % Update c
    c = [Gamma_Mstar*(1+h_Mstar*fn); Kappa_Gstar; Kappa_PLCstar; Gamma_GAP; 
    Gamma_G; Kappa_Dstar; Gamma_PLCstar*(1+h_PLCstar*fn); 
    Gamma_Dstar*(1+h_Dstar*fn); Kappa_Tstar*(1+h_TstarP*fp)/Kappa_Dstar^2;
    Gamma_Tstar*(1+h_TstarN*fn); K_u; K_r];
        
    % Update Current for Calcium
    I_in = [I_in I_Tstar*X{7}(i)];      
    I_Ca = P_Ca * I_in(i);         
    I_NaCa = K_NaCa*(((Na_i^3)*(Ca_o^2)) - ((Na_o^3)*Ca2(i-1)*exp(-V_m*F/R/T)));
    I_Canet = I_Ca + 2*I_NaCa;
    
    % Update Calcium
    Ca2 = [Ca2 ((I_Canet/2/v/F) + n*K_r*C_star_conc + f1)/(n*K_u*(C_T - C_star_conc) + K_Ca + f2)];%#ok<*AGROW>
        
    if Ca2(i) < 0
        Ca2(i) = 0;
    end
    
    LIC = I_in(i)*30000/(10^6)/(1.57e-5);
    
    z = F_HHN([V_m sa si dra dri],LIC);
    
    V_m = V_m + z(1)*dt;
    sa = sa + z(2)*dt;
    si = si + z(3)*dt;
    dra = dra + z(4)*dt;
    dri = dri + z(5)*dt;
    
    f2 = K_NaCa*(exp(-V_m*F/R/T))*((Na_o)^3)/v/F;  % Update f2 (based on membrane voltage)

    
    
    
    
end
%end
    


figure('Position',[10 30 600 400]);
subplot(421);
plot(X{1});
xlim([0 1000]);
ylim([0 1]);
title('M*');

subplot(423);
plot(X{2});
title('G');
%ylim([0 50]);
xlim([0 1000]);

subplot(425);
plot(X{3});
title('G*');
%ylim([0 1]);
xlim([0 1000]);

subplot(427);
plot(X{4});
title('PLC*');
%ylim([0 2]);
xlim([0 1000]);

subplot(422);
plot(X{5});
title('D*');
%ylim([0 30]);
xlim([0 1000]);

subplot(424);
plot(X{6});
title('C*');
%ylim([0 60]);
xlim([0 1000]);

subplot(426);
plot(X{7});
title('T*');
%ylim([0 25]);
xlim([0 1000]);

subplot(428);
plot(Ca2);
title('Ca(2+)');
%ylim([0 0.8]);
xlim([0 1000]);


