% Function RPAM takes in a number of photons (per ms) and outputs
% a Poisson distribution of activated microvilli (out of 30,000).
% N(1) = # of activated microvilli that absorbed 0 photons, N(2) absorbed 1
% photon, etc. etc.

function Np = RPAM(Nphoton)

if (Nphoton > 1000)
    Nphoton = 1000;
elseif (Nphoton < 100)
    Nphoton = 100;
end

Nmicro = 30000;                 % Number of microvilli
Nm  = Nmicro;                   % Number of activated microvilli
Np = zeros(Nmicro,1);           % Number of photons captured by each microvillus
fe = 0;                         % Fraction of microvilli that escape activation (Zero photons absorbed)
fa = 0;                         % Fraction of microvilli that are activated (1 - fe)
LambdaM = 0;                    % Average number of photons absorbed per microvillus
x = [0 1 2 3 4 5];              % Possible number of photons absorbed
fx = zeros(6,1);                % Fraction of microvilli that absorb x photons

flag = true;                    % Boolean flag
while (flag == true)
    
    LambdaM = Nphoton/Nm;  
    
    for ii = 1:length(x)        % Calculate each number of microvilli with activated photons
        fx(ii) = (exp(-LambdaM))*power(LambdaM,x(ii))/factorial(x(ii));
    end
    
    fe = exp(-LambdaM);
    fa = 1 - fe;                % Percent of activated photons
    
    Nmtemp = Nmicro*fa;         % Number of activated photons
    
    if (abs(Nmtemp - Nm) < 1)   % If Nm is converging
        flag = false;
    end
    
    Nm = Nmtemp;
     
end

LambdaP = Nphoton/Nm;
km = 10*round(LambdaP + 1);
p = zeros(km,1);
q = zeros(km,1);

for i = 1:km
    p(i) = exp(-LambdaP) * power(LambdaP,(i-1)) / factorial(i-1);
end
    
for i = 2:km
    q(i) = sum(p(2:i))/(sum(p)-p(1));
end

r = zeros(1,Nmicro);

for n = 1:Nmicro
    r(n) = rand;

    for i = 1:length(q)
        if r(n) < q(i)
            Np(n) = i - 2;
            break;
        end
    end
end

N = zeros(1,4);

for n = 1:Nm
    for m = 0:3
        if Np(n) == m 
            N(m+1) = N(m+1) + 1;
        end
    end
end

%bar(N(2:end));


end
