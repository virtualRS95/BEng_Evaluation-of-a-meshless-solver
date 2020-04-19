clear

AR = 6.7;
c = 1;
K0 = 2*pi;
n = 4;
alpha_abs = 5*pi/180; %degrees
U=1;


% arbitrary rectangular wing
% pre-allocation
theta0 = zeros(n,1);
M = zeros(n);

% calculate
for m = 1:n
   theta0(m)= pi/2-(m-1)*pi/2/n; %root to outboard
   for j = 1:n
       M(m,j)= sin((2*j-1)*theta0(m)) + K0/(4*AR)*(2*j-1)*sin((2*j-1)*theta0(m))/sin(theta0(m));
   end
end

alpha = alpha_abs*ones(n,1);
A = inv(M)*alpha;

CL = K0*pi*A(1)/4


summation = 0;
for j = 1:n
    summation = summation + j*A(j);
end
CDi = K0^2*pi/16/AR*summation


CDi_el = CL^2/pi/AR
factor = 1;
for j = 2:n
    factor = factor + j*A(j)^2/A(1)^2;
end
CDi = CDi_el*factor

sA=0;
for j =1:n
    if rem(j,2)==0
        sA=sA+A(j);
    else
        sA=sA-A(j);
    end
end
gamma = K0*c*U/2*sA