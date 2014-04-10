%{
dt = 1e-3;
t = 0:dt:1;

NT = zeros(30000,length(t));
Nphoton = 1000;

for i = 1:length(t)
    NT(:,i) = RPAM(Nphoton);
end
%}


n = 1;
X = Signal_Cascade(NT(n,:));



