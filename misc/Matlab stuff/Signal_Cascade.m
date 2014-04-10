function X = Signal_Cascade(N_ph)

%% Initialize Parameters

% la between 0.1 and 1
% ns = 1 if dim and 2 if bright

la = 0.5;
n_s = 2;
parameters;

%% X initialization

X = zeros(7, length(N_ph));

X(1,1) = N_ph(1);
X(2,1)= 50;
X(3,1) = 0;
X(4,1) = 0;
X(5,1) = 0;
X(6,1) = 0;
X(7,1) = 0;

Ca2 = rand; 

% Hill functions for positive and negative feedback
fp = ((Ca2/K_p)^m_p)/(1+(Ca2/K_p)^m_p);
fn = n_s*((X(6)/K_n)^m_n)/(1+(X(6)/K_n)^m_n);

% Transition matrix
V = zeros(7,12);
V(1,1) = -1;
V(2,2) = -1;
V(2,5) = 1;
V(3,2) = 1;
V(3,3) = -1;
V(3,4) = -1;
V(4,3) = 1;
V(4,7) = -1;
V(5,6) = 1;
V(5,8) = -1;
V(5,9) = -2;
V(6,11) = 1;
V(6,12) = -1;
V(7,9) = 1;
V(7,10) = -1;

h = [X(1,1); X(1,1)*X(2,1); X(3,1)*(PLC_T - X(4,1)); X(3,1)*X(4,1); 
    (G_T - X(3,1) - X(2,1) - X(4,1)); X(4,1); X(4,1); X(5,1); 
    .5*(X(5,1)*(X(5,1)-1)*(T_T-X(7,1))); X(7,1); C_T - X(6,1); X(6,1)];

c = [Gamma_Mstar*(1+h_Mstar*fn); Kappa_Gstar; Kappa_PLCstar; Gamma_GAP; 
    Gamma_G; Kappa_Dstar; Gamma_PLCstar*(1+h_PLCstar*fn); 
    Gamma_Dstar*(1+h_Dstar*fn); Kappa_Tstar*(1+h_TstarP*fp)/Kappa_Dstar^2;
    Gamma_Tstar*(1+h_TstarN*fn); K_u/v^2; K_r];

I_in = I_Tstar * X(7,1);      % Initialize input current (X(7,1) is Tstar)
I_Ca = P_Ca * I_in;         % Calcium Current ~40%
I_NaCa = K_NaCa*(((Na_i^3)*(Ca_o^2)) - ((Na_o^3)*Ca2*exp(V_m*F/R/T)));
I_Canet = I_Ca - 2*I_NaCa;

% To establish time vector of where activations occur
T_ph = [];
for n = 1:length(N_ph)
    if N_ph(n) ~= 0
        T_ph = [T_ph n*1e-3];
    end
end 

i = 1;                      % Iterator
ii = 1;                     % X iterator
t = 0;                      % Time
tend = 1;                   % End time
mu = 1;                     % Reaction to perform each dt
av = zeros(1,7);            % For determining mu

%% Phototransduction
while (t < tend)
    
    ii = ii + 1;
    
    % Determine stochastic time step
    r1 = rand;
    r2 = rand;
    
    as = dot(h,c);
    dt = (1/(la + as)) * log(1/r1);
    
    % If our random timestep includes a photon...
    if ((t + dt) > T_ph(i))
        t = T_ph(i);
        i = i + 1;
        X(1,ii) = X(1,ii-1) + N_ph(t*1000);  
    else
        t = t + dt;
    end
    
    % To determine which reaction to perform in this time step
    av(1) = dot(h(1:1),c(1:1));
    
    for k = 2:length(av)
        
        av(k) = dot(h(1:k),c(1:k));
        
        if ((r2*as > av(k-1)) && (r2*as <= av(k)))
            mu = k;
        end
        
    end
    
    % Update X(mu)
    temp = 0;
    
    for m = 1:12
        temp = temp + V(mu,m)*h(m)*c(m);
    end
    
    if mu == 1
        X(mu,ii) = X(mu,ii) + temp;
    else
        X(mu,ii) = X(mu,ii-1) + temp;
    end
    
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