function [R] = Metodo_Euler(ro,vo,mu,tf,N)

h = 1./(2.^N);  %Delta de integracion
k = 1;

%Integracion
    
%Posicion y velocidad inicial
r = ro;
v = vo;    
R(1,:) = ro;
V(1,:) = vo;
    
for i = 1:tf/h
    r = r + v*(h^k);
    v = v - mu*r*(h^k)/(norm(r))^3;
    R(i+1,:) = r;
    V(i+1,:) = v;
end

end