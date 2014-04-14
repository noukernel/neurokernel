function [X, t, abs] = Signal_Cascade(X_old, t, T_ph, N_ph, h, c)

mu = 1;                     % Reaction to perform each dt
av = zeros(1,7);            % For determining mu
la = 0.5;
n_s = 2;
parameters;

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

%% Phototransduction
X = X_old;

% Determine stochastic time step
r1 = rand;
    
as = dot(h,c);
dt = (1/(la + as)) * log(1/r1);
    
% If our random timestep includes a photon...
if ((t + dt) > T_ph)
    t = T_ph;
    X(1) = X(1) + N_ph;
    abs = true;
else
    t = t + dt;
    abs = false;
end
    
r2 = rand;

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
    
for z = 1:7
    if z == mu
        X(z) = X(z) + temp;
    else
        X(z) = X(z);
    end
    
    if X(z) < 0
        X(z) = 0;
    end
end



end