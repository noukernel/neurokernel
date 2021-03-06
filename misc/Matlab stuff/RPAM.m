function Np = RPAM(num_photons)

Nmicro = 30000;
Nph = num_photons;
LambdaM = Nph/Nmicro;
k = -1;
flag = true;

while (flag)
    k = k + 1;

    p(k+1) = (LambdaM^k)*exp(-LambdaM)/factorial(k);
    
    if p(k+1) < (1/Nmicro)
        flag = false;
    end

end

p = round(p*Nmicro);
p(1) = 0;


%r = 1:30000;
r = randperm(30000);

Np = zeros(1,30000);

for ii = 2:length(p)
    for jj = 1:p(ii)
        Np(r(jj+p(ii-1))) = ii - 1;
    end
end

%bar(N(2:end));
    
