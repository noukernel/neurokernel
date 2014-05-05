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
ddt = 1e-4;
tend = 1;

la = 0.5;
n_s = 2;

NA = 6.02e23;
v = 3e-9;

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
C_T_conc = 0.5;
C_T_num = C_T_conc*NA*v*10^(-12); %903
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
K_NaCa = 3e-8;
C_m = 62.8;
V_m = -70e-3;
f1 = K_NaCa*((Na_i)^3)*((Ca_o)^2)/v/F;
f2 = K_NaCa*(exp(-V_m*F/R/T))*((Na_o)^3)/v/F;
sa = 0.6982;
si = 6.6517e-5;
dra = 0.2285;
dri = 1.2048e-4;

%% Initialization of Matrices

n_m = 30;

X = cell(7, n_m);
Ca2 = zeros(1,n_m);
mu = zeros(1,1);
I_in = zeros(1,n_m);
print_V = [];
print_I = [];
print_X = [];
print_Ca = [];
for runtime = 1:200
%for ii = 1:30000

LIC = zeros(1,n_m);
for ii = 1:n_m

    % init code
    if runtime == 1
        for m = 1:7
            X{m,ii} = zeros(1,1);
        end
        
        X{1,ii} = 1;% Undefined:t+ Np(1);
        %X{1} = zeros(1,1001);
        %X{1}(1:100) = 1;
        X{2,ii} = 50;
        X{3,ii} = 0;
        X{4,ii} = 0;
        X{5,ii} = 0;
        X{6,ii} = 0;
        X{7,ii} = 0;

        Ca2(1,ii) = Ca_dark; 
        C_star_conc = X{6,ii}/v/NA*10^12;
        C_star_num = X{6,ii};
        CaM_num = C_T_num;
        CaM_conc = C_T_conc;
    

        % Hill functions for positive and negative calcium feedback
        fp = ((Ca2(1,ii)/K_p)^m_p)/(1+(Ca2(1,ii)/K_p)^m_p);
        fn = n_s*((C_star_conc/K_n)^m_n)/(1+(C_star_conc/K_n)^m_n);

        h = [X{1,ii}; X{1,ii}*X{2,ii}; X{3,ii}*(PLC_T - X{4,ii}); X{3,ii}*X{4,ii}; 
        (G_T - X{3,ii} - X{2,ii} - X{4,ii}); X{4,ii}; X{4,ii}; X{5,ii}; 
        .5*(X{5,ii}*(X{5,ii}-1)*(T_T-X{7,ii})); X{7,ii}; Ca2(1,ii)*CaM_num; X{6,ii}];

        c = [Gamma_Mstar*(1+h_Mstar*fn); 
            Kappa_Gstar; 
            Kappa_PLCstar; 
            Gamma_GAP; 
            Gamma_G; 
            Kappa_Dstar; 
            Gamma_PLCstar*(1+h_PLCstar*fn); 
            Gamma_Dstar*(1+h_Dstar*fn); 
            Kappa_Tstar*(1+h_TstarP*fp)/K_Dstar^2;
            Gamma_Tstar*(1+h_TstarN*fn); 
            K_u; 
            K_r];


        I_in(1,ii) = I_Tstar * X{7,ii};      % Initialize input current (X(7,1) is Tstar)
        I_Ca = P_Ca * I_in(1,ii);         % Calcium Current ~40%
        I_NaCa = K_NaCa*(((Na_i^3)*(Ca_o)) - ((Na_o^3)*Ca2(1,ii)*exp(-V_m*F/R/T)));
        I_Canet = I_Ca + 2*I_NaCa;
    end

    i = 1;          % X iterator (how many times Signal_Cascade was called)
    %ii = 1;         % T_ph iterator (next photon absorption)
    %tt = 0;         % Timekeeper
    t = 0;
    %For one microvillus
    % when this is a script, it resets X for some reason.
    % Memory allocation problem? Very strange.
    
    Np = RPAM(100 + round(rand(1)*900)); % random num between 100 and 1000
    X{1,ii} = Np(ii);
    %{
    if (runtime < 20)
        X{1,ii} = 1;
    else
        X{1,ii} = 0;
    end
    %}
    while (t < ddt)
    
        X_old = [X{1,ii}, X{2,ii}, X{3,ii}, X{4,ii}, X{5,ii}, X{6,ii}, X{7,ii}];
    
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
    
        for m = 1:7
        %for m = 2:7
            X{m,ii} = Z(m);%#ok<*AGROW>
        end
    
        C_star_conc = X{6,ii}/v/NA*10^12;
        C_star_num = X{6,ii};
        CaM_num = C_T_num - C_star_num;
        CaM_conc = CaM_num/v/NA*10^12;
        
        % Update h
        h = [X{1,ii}; X{1,ii}*X{2,ii}; X{3,ii}*(PLC_T - X{4,ii}); X{3,ii}*X{4,ii}; 
        (G_T - X{3,ii} - X{2,ii} - X{4,ii}); X{4,ii}; X{4,ii}; X{5,ii}; 
        .5*(X{5,ii}*(X{5,ii}-1)*(T_T-X{7,ii})); X{7,ii}; Ca2(1,ii)*CaM_num; X{6,ii}];
    
        % Update fp, fn
        fp = ((Ca2(1,ii)/K_p)^m_p)/(1+(Ca2(1,ii)/K_p)^m_p);
        fn = n_s*((C_star_conc/K_n)^m_n)/(1+(C_star_conc/K_n)^m_n);
    
        % Update c
        c = [Gamma_Mstar*(1+h_Mstar*fn); Kappa_Gstar; Kappa_PLCstar; Gamma_GAP; 
        Gamma_G; Kappa_Dstar; Gamma_PLCstar*(1+h_PLCstar*fn); 
        Gamma_Dstar*(1+h_Dstar*fn); Kappa_Tstar*(1+h_TstarP*fp)/K_Dstar^2;
        Gamma_Tstar*(1+h_TstarN*fn); K_u; K_r];
        
        % Update Current for Calcium
        I_in(1,ii) = I_Tstar*X{7,ii};      
        I_Ca = P_Ca * I_in(1,ii);         
        I_NaCa = K_NaCa*(((Na_i^3)*(Ca_o)) - ((Na_o^3)*Ca2(1,ii)*exp(-V_m*F/R/T)));
        I_Canet = I_Ca + 2*I_NaCa;
    
        % Update Calcium
        %Ca2(1,ii) = ((I_Canet/2/v/F) + n*K_r*C_star_conc + f1)/(n*K_u*(C_T - C_star_conc) + K_Ca + f2);%#ok<*AGROW>
        Ca2(1,ii) = ((I_Canet/2/v/F) + n*K_r*C_star_conc + f1)/(n*K_u*CaM_conc + K_Ca + f2);%#ok<*AGROW>
        
        if Ca2(1,ii) < 0
            Ca2(1,ii) = 0;
        end
    
        LIC(ii) = LIC(ii) + I_in(1,ii);%/(10^6)/(1.57e-5);
    end

end

    LIC_tot = sum(LIC)/ 1e6 / 1.57e-5;
    %LIC_tot = 0;
    %end
    for i = 1:10
        dddt = ddt/10;
        
        z = F_HHN([V_m*1000, sa, si, dra, dri], LIC_tot);
    
        V_m = V_m + z(1)*dddt;
        sa = sa + z(2)*dddt;
        si = si + z(3)*dddt;
        dra = dra + z(4)*dddt;
        dri = dri + z(5)*dddt;
    
        print_V = [print_V V_m];
    end
    
    print_I = [print_I LIC_tot];
    print_Ca = [print_Ca Ca2(1)];
    
    % get a sample of the microvilli
    temp_x = [X{1,1};X{2,1};X{3,1};X{4,1};X{5,1};X{6,1};X{7,1};]; 
    print_X = [print_X temp_x];
    
    f2 = K_NaCa*(exp(-V_m*F/R/T))*((Na_o)^3)/v/F;  % Update f2 (based on membrane voltage)
end
    


figure('Position',[10 30 600 400]);
subplot(421);
plot(print_X(1,:));
xlim([0 runtime]);
ylim([0 2]);
title('M*');

subplot(423);
plot(print_X(2,:));
title('G');
ylim([0 60]);
xlim([0 runtime]);

subplot(425);
plot(print_X(3,:));
title('G*');
ylim([0 3]);
xlim([0 runtime]);

subplot(427);
plot(print_X(4,:));
title('PLC*');
ylim([0 3]);
xlim([0 runtime]);

subplot(422);
plot(print_X(5,:));
title('D*');
%ylim([0 30]);
xlim([0 runtime]);

subplot(424);
plot(print_X(6,:));
title('C*');
%ylim([0 60]);
xlim([0 runtime]);

subplot(426);
plot(print_X(7,:));
title('T*');
ylim([0 5]);
xlim([0 runtime]);

subplot(428);
plot(print_Ca);
title('Ca(2+)');
%ylim([0 2]);
xlim([0 runtime]);

figure;
%subplot(211);
plot(print_V);

figure;
%subplot(212);
plot(print_I);
