%% Random Photon Absorption Model
%{
dt = 1;
t = 0:dt:1; % Only doing 1 'time step'
tend = max(t);

NT = zeros(30000,length(t));
Nphoton = 1000;


NT(:,1) = RPAM(Nphoton);

% Should we store separate I_in per microvillus? No idea...
I_in = zeros(30000,1);

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
n_s = 1;
parameters;

%% Initialization of Matrices

X = cell(7,1);

%for ii = 1:30000

for m = 1:7
    X{m} = zeros(1,1);
end

X{1}(1) = 1; %NT(ii,1);
X{2}(1) = 50;
X{3}(1) = 0;
X{4}(1) = 0;
X{5}(1) = 0;
X{6}(1) = 0;
X{7}(1) = 0;

Ca2 = 160e-6; 

% Hill functions for positive and negative calcium feedback
fp = ((Ca2/K_p)^m_p)/(1+(Ca2/K_p)^m_p);
fn = n_s*((X{6}(1)/K_n)^m_n)/(1+(X{6}(1)/K_n)^m_n);

h = [X{1}(1); X{1}(1)*X{2}(1); X{3}(1)*(PLC_T - X{4}(1)); X{3}(1)*X{4}(1); 
    (G_T - X{3}(1) - X{2}(1) - X{4}(1)); X{4}(1); X{4}(1); X{5}(1); 
    .5*(X{5}(1)*(X{5}(1)-1)*(T_T-X{7}(1))); X{7}(1); Ca2*(C_T - X{6}(1)); X{6}(1)];

c = [Gamma_Mstar*(1+h_Mstar*fn); Kappa_Gstar; Kappa_PLCstar; Gamma_GAP; 
    Gamma_G; Kappa_Dstar; Gamma_PLCstar*(1+h_PLCstar*fn); 
    Gamma_Dstar*(1+h_Dstar*fn); Kappa_Tstar*(1+h_TstarP*fp)/Kappa_Dstar^2;
    Gamma_Tstar*(1+h_TstarN*fn); K_u; K_r];

I_in = I_Tstar * X{7}(1);      % Initialize input current (X(7,1) is Tstar)
I_Ca = P_Ca * I_in;         % Calcium Current ~40%
I_NaCa = K_NaCa*(((Na_i^3)*(Ca_o^2)) - ((Na_o^3)*Ca2*exp(-V_m*F/R/T)));
I_Canet = I_Ca - 2*I_NaCa;

i = 1;          % X iterator (how many times Signal_Cascade was called)
%ii = 1;         % T_ph iterator (next photon absorption)
%tt = 0;         % Timekeeper

%For one microvillus
while t < tend
    
    X_old = [X{1}(i), X{2}(i), X{3}(i), X{4}(i), X{5}(i), X{6}(i), X{7}(i)];
    
    [Z, t] = Signal_Cascade(X_old, t, h, c);
    
    %tt = tt + 1;%t_new;
    
    % No idea, but won't matter for now
    %{
    if abs == true
        ii = ii + 1;
    end
    %}
    
    i = i + 1;
    
    for m = 1:7
        X{m} = [X{m} Z(m)];
    end
        
    % Update Current for Calcium
    I_in = I_Tstar * X{7}(i);      
    I_Ca = P_Ca * I_in;         
    I_NaCa = K_NaCa*(((Na_i^3)*(Ca_o^2)) - ((Na_o^3)*Ca2*exp(V_m*F/R/T)));
    I_Canet = I_Ca + 2*I_NaCa;
    
    % Update Calcium
    Ca2 = v*((I_Canet/2/v/F) + n*K_r*X{6}(i) + f1)/(n*K_u*(C_T - X{6}(i)) + K_Ca + f2);
    
    % Update h
    h = [X{1}(i); X{1}(i)*X{2}(i); X{3}(i)*(PLC_T - X{4}(i)); X{3}(i)*X{4}(i); 
    (G_T - X{3}(i) - X{2}(i) - X{4}(i)); X{4}(i); X{4}(i); X{5}(i); 
    .5*(X{5}(i)*(X{5}(i)-1)*(T_T-X{7}(i))); X{7}(i); C_T - X{6}(i); X{6}(i)];
    
    % Update fp, fn
    fp = ((Ca2/K_p)^m_p)/(1+(Ca2/K_p)^m_p);
    fn = n_s*((X{6}(i)/K_n)^m_n)/(1+(X{6}(i)/K_n)^m_n);
    
    % Update c
    c = [Gamma_Mstar*(1+h_Mstar*fn); Kappa_Gstar; Kappa_PLCstar; Gamma_GAP; Gamma_G; Kappa_Dstar;
    Gamma_PLCstar*(1+h_PLCstar*fn); Gamma_Dstar*(1+h_Dstar*fn); Kappa_Tstar*(1+h_TstarP*fp)/Kappa_Dstar^2;
    Gamma_Tstar*(1+h_TstarN*fn); K_u/v^2; K_r];
    
end
%end
    
%{ 
Working on this part now    
for mv = 1:30000
    
    while tt <= tend
        
        Xn = Signal_Cascade(X, NT(mv,tt), tt, h, c, as, dt);
    
        for m = 1:7
            X{m} = [X{m} Xn(m)];
        end
        
        tt = tt + 1;
    
    
    
    
    
    end 
    
end

%}

figure;
subplot(811);
plot(X{1});
subplot(812);
plot(X{2});
subplot(813);
plot(X{3});
subplot(814);
plot(X{4});
subplot(815);
plot(X{5});
subplot(816);
plot(X{6});
subplot(817);
plot(X{7});
subplot(818);
plot(Ca2);



