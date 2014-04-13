%% Random Photon Absorption Model

dt = 1e-3;
t = 0:dt:1;
tend = max(t);

NT = zeros(30000,length(t));
Nphoton = 1000;

for i = 1:length(t)
    NT(:,i) = RPAM(Nphoton);
end


%% Initialize Parameters

% la between 0.1 and 1
% ns = 1 if dim and 2 if bright

la = 0.5;
n_s = 2;
parameters;

%% Initialization of Matrices

X = cell(7,1);

for m = 1:7
    X{m} = zeros(1,1);
end

X{1}(1) = 4;
X{2}(1) = 50;
X{3}(1) = 2;
X{4}(1) = 6;
X{5}(1) = 2;
X{6}(1) = 1;
X{7}(1) = 1;

Ca2 = rand; 

% Hill functions for positive and negative feedback
fp = ((Ca2/K_p)^m_p)/(1+(Ca2/K_p)^m_p);
fn = n_s*((X{6}(1)/K_n)^m_n)/(1+(X{6}(1)/K_n)^m_n);

h = [X{1}(1); X{1}(1)*X{2}(1); X{3}(1)*(PLC_T - X{4}(1)); X{3}(1)*X{4}(1); 
    (G_T - X{3}(1) - X{2}(1) - X{4}(1)); X{4}(1); X{4}(1); X{5}(1); 
    .5*(X{5}(1)*(X{5}(1)-1)*(T_T-X{7}(1))); X{7}(1); C_T - X{6}(1); X{6}(1)];

c = [Gamma_Mstar*(1+h_Mstar*fn); Kappa_Gstar; Kappa_PLCstar; Gamma_GAP; 
    Gamma_G; Kappa_Dstar; Gamma_PLCstar*(1+h_PLCstar*fn); 
    Gamma_Dstar*(1+h_Dstar*fn); Kappa_Tstar*(1+h_TstarP*fp)/Kappa_Dstar^2;
    Gamma_Tstar*(1+h_TstarN*fn); K_u; K_r];

I_in = I_Tstar * X{7}(1);      % Initialize input current (X(7,1) is Tstar)
I_Ca = P_Ca * I_in;         % Calcium Current ~40%
I_NaCa = K_NaCa*(((Na_i^3)*(Ca_o^2)) - ((Na_o^3)*Ca2*exp(V_m*F/R/T)));
I_Canet = I_Ca - 2*I_NaCa;

tt = 1;
mu = 1;                     % Reaction to perform each dt
av = zeros(1,7);            % For determining mu


[Zn, zn] = Signal_Cascade(Z,0,1,.001,h,c);
    
% Working on this part now    
for mv = 1:30000
    
    while tt <= tend
        
        Xn = Signal_Cascade(X, NT(mv,tt), tt, h, c, as, dt);
    
        for m = 1:7
            X{m} = [X{m} Xn(m)];
        end
        
        tt = tt + 1;
    
    
    % Update Current for Calcium
    I_in = I_Tstar * X(7,ii);      
    I_Ca = P_Ca * I_in;         
    I_NaCa = K_NaCa*(((Na_i^3)*(Ca_o^2)) - ((Na_o^3)*Ca2*exp(V_m*F/R/T)));
    I_Canet = I_Ca - 2*I_NaCa;  % Not sure we need this one
    
    % Update Calcium
    % Not sure if this is the way to do it, or use the differential 
    % eq for Calcium (eq. 39 in 2012 paper)
    Ca2 = v*((I_Ca/2/v/F) + n*K_r*X(6,ii) - f1)/(n*K_u*(C_T - X(6,ii)) + K_Ca - f2);
    
    % Update h
    h = [X(1,ii); X(1,ii)*X(2,ii); X(3,ii)*(PLC_T - X(4,ii)); X(3,ii)*X(4,ii); 
        (G_T - X(3,ii) - X(2,ii) - X(4,ii)); X(4,ii); X(4,ii); X(5,ii); 
        .5*(X(5,ii)*(X(5,ii)-1)*(T_T-X(7,ii))); X(7,ii); C_T - X(6,ii); X(6,ii)];
    
    % Update fp, fn
    fp = ((Ca2/K_p)^m_p)/(1+(Ca2/K_p)^m_p);
    fn = n_s*((X(6,ii)/K_n)^m_n)/(1+(X(6,ii)/K_n)^m_n);
    
    % Update c
    c = [Gamma_Mstar*(1+h_Mstar*fn); Kappa_Gstar; Kappa_PLCstar; Gamma_GAP; Gamma_G; Kappa_Dstar;
    Gamma_PLCstar*(1+h_PLCstar*fn); Gamma_Dstar*(1+h_Dstar*fn); Kappa_Tstar*(1+h_TstarP*fp)/Kappa_Dstar^2;
    Gamma_Tstar*(1+h_TstarN*fn); K_u/v^2; K_r];
    
    
    end 
    
end
%}




