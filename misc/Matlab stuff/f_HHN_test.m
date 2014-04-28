dt = 1e-5;
t = 0:dt:1;
V = [-0.070];
sa = [0];
si = [0];
dri = [0];
dra = [0];
I = 0.6;


for ii = 1:length(t)
    x = F_HHN([V(ii), sa(ii), si(ii), dri(ii), dra(ii)], I);
    V  = [V (V(ii) + x(1) * dt)];
    sa  = [sa (sa(ii) + x(2) * dt)];
    si  = [si (si(ii) + x(3) * dt)];
    dri  = [dri (dri(ii) + x(4) * dt)];
    dra  = [dra (dra(ii) + x(5) * dt)];
end