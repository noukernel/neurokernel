function [X, dt] = Signal_Cascade(X_old, h, c)

mu = 1;                     % Reaction to perform each dt
av = zeros(1,12);            % For determining mu
la = 0.5;
parameters;

%% Phototransduction
X = X_old;

% Determine stochastic time step
r1 = rand;
    
as = dot(h,c);
dt = (1/(la + as)) * log(1/r1);

%{    
% If our random timestep includes a photon...
if ((t + dt) > T_ph)
    t = T_ph;
    X(1) = X(1) + N_ph;
    abs = true;
else
    t = t + dt;
    abs = false;
end
%}   

%t = t + dt;


r2 = rand;

% To determine which reaction to perform in this time step
av(1) = dot(h(1:1),c(1:1)); 
for k = 2:length(av)
        
    av(k) = dot(h(1:k),c(1:k));
        
    if ((r2*as > av(k-1)) && (r2*as <= av(k)))
        mu = k;
    end
        
end

zz = av';

% Update X(mu)
hc = zeros(1,12);
for m = 1:12
    hc(m) = h(m)*c(m);
end

%V Transition Matrix
if(mu == 1) 
    X(1) = X(1) - 1;
elseif (mu == 2)
    X(2) = X(2) - 1;
    X(3) = X(3) + 1;
elseif (mu == 3)
    X(3) = X(3) - 1;
    X(4) = X(4) + 1;
elseif (mu == 4)
    X(3) = X(3) - 1;
elseif (mu == 5)
    X(2) = X(2) + 1;
elseif (mu == 6)
    X(5) = X(5) + 1;
elseif (mu == 7)
    X(4) = X(4) - 1;
elseif (mu == 8)
    X(5) = X(5) - 1;
elseif (mu == 9)
    X(5) = X(5) - 2;
    X(7) = X(7) + 1;
elseif (mu == 10)
    X(7) = X(7) - 1;
elseif (mu == 11)
    X(6) = X(6) + 1;
else
    X(6) = X(6) - 1;
end

for m = 1:7
    if X(m) < 0
        X(m) = 0;
    end
end


end