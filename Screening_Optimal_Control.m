clear; clc; close all;

% Call the function 'screening_control' with parameters and store the output in 'y'
y = screening_control(100,9.01623,0.556, 0.5, 0.014156,0.11, 0.487, 0.2, 0.77, 0.104, 0.68);

function y = screening_control(T, Lambda, b, delt, u,u3, gam, e, r, the, tel)

% Define a convergence test variable (initially set to -1)
test = -1;

% Define delta for termination criteria
delta = 0.5;

% Define number of grid points
M = 99;

% Create a time grid from 1 to T with M+1 points
t=linspace(1,T,M+1);

h=T/(M+1);
h2 = h/2;

% Define model parameters - Cost of Cancer treatment (A) and cost per one
% screening test (b)
A = 60000;
B = 500;

% Initialize state and co-state variables
S=zeros(1,M+1);
H=zeros(1,M+1);
I=zeros(1,M+1);
C=zeros(1,M+1);
R=zeros(1,M+1);
no=ones(1,M+1);

% Set initial conditions for state variables
x0 = [422.00000,0,29.820000, 0.099100, 0];

S(1)=x0(1);
H(1) = x0(2);
I(1)=x0(3);
C(1)=x0(4);
R(1)=x0(5);

lambda1=zeros(1,M+1);
lambda2=zeros(1,M+1);
lambda3=zeros(1,M+1);
lambda4=zeros(1,M+1);
lambda5=zeros(1,M+1);

a = zeros(1,M+1);

while(test < 0)
    
    % Store old values of states and co-states for convergence test
    olda = a;
    oldS = S;
    oldH = H;
    oldI = I;
    oldC = C;
    oldR = R;

    oldlambda1 = lambda1;
    oldlambda2 = lambda2;
    oldlambda3 = lambda3;
    oldlambda4 = lambda4;
    oldlambda5 = lambda5;
    
    % Loop for forward sweep (calculate states)
    for i = 1:M
        no(i) = S(i) + I(i) + H(i) + C(i) + R(i);

        % Calculate right-hand side of state equations using forward difference and current co-state values
        m11 = Lambda + (e * R(i)) - 0.103*a(i)*S(i) - (1-a(i))*b*S(i)*I(i)/no(i) - u*S(i);
        m12 = b *(1-a(i))*S(i)*I(i)/no(i)  - (delt + u + gam) * I(i);
        m13 = 0.103*a(i)*S(i) - (the + tel + u) * H(i);
        m14 = the*H(i) + gam*I(i) - (r + u + u3)*C(i);
        m15 = delt * I(i) + tel * H(i) + r * C(i) - (e + u) * R(i);

        m21 = Lambda + (e * (R(i)+h2*m15)) - 0.103*a(i)*(S(i) + h2*m11) - (1-a(i))*b*(S(i) + h2*m11)*(I(i) + h2*m12)/no(i) - u*(S(i) + h2*m11);
        m22 = b *(1-a(i))*(S(i)+ h2*m11)*(I(i) + h2*m12)/no(i)  - (delt + u + gam) * (I(i)+h2*m12);
        m23 = 0.103*a(i)*(S(i) + h2*m11) - (the + tel + u) * (H(i) + h2*m13);
        m24 = the*(H(i) + h2*m13) + gam*(I(i) + h2*m12) - (r + u + u3)*(C(i) + h2*m14);
        m25 = delt * (I(i) + h2*m12) + tel * (H(i) + h2*m13) + r * (C(i) + h2*m14) - (e + u) * (R(i) + h2*m15);
        
        m31 = Lambda + (e * (R(i)+h2*m25)) - 0.103*a(i)*(S(i) + h2*m21) - (1-a(i))*b*(S(i) + h2*m21)*(I(i) + h2*m22)/no(i) - u*(S(i) + h2*m21);
        m32 = b *(1-a(i))*(S(i)+ h2*m21)*(I(i) + h2*m22)/no(i)  - (delt + u + gam) * (I(i)+h2*m22);
        m33 = 0.103*a(i)*(S(i) + h2*m21) - (the + tel + u) * (H(i) + h2*m23);
        m34 = the*(H(i) + h2*m23) + gam*(I(i) + h2*m22) - (r + u + u3)*(C(i) + h2*m24);
        m35 = delt * (I(i) + h2*m22) + tel * (H(i) + h2*m23) + r * (C(i) + h2*m24) - (e + u) * (R(i) + h2*m25);
        
        m41 = Lambda + (e * (R(i)+h2*m35)) - 0.103*a(i)*(S(i) + h2*m31) - (1-a(i))*b*(S(i) + h2*m31)*(I(i) + h2*m32)/no(i) - u*(S(i) + h2*m31);
        m42 = b *(1-a(i))*(S(i)+ h2*m31)*(I(i) + h2*m32)/no(i)  - (delt + u + gam) * (I(i)+h2*m32);
        m43 = 0.103*a(i)*(S(i) + h2*m31) - (the + tel + u) * (H(i) + h2*m33);
        m44 = the*(H(i) + h2*m33) + gam*(I(i) + h2*m32) - (r + u + u3)*(C(i) + h2*m34);
        m45 = delt * (I(i) + h2*m32) + tel * (H(i) + h2*m33) + r * (C(i) + h2*m34) - (e + u) * (R(i) + h2*m35);
        
        % Update states using 4th order Runge-Kutta approximation
        S(i+1) = S(i) + (h/6)*(m11 + 2*m21 + 2*m31 + m41);
        I(i+1) = I(i) + (h/6)*(m12 + 2*m22 + 2*m32 + m42);
        H(i+1) = H(i) + (h/6)*(m13 + 2*m23 + 2*m33 + m43);
        C(i+1) = C(i) + (h/6)*(m14 + 2*m24 + 2*m34 + m44);
        R(i+1) = R(i) + (h/6)*(m15 + 2*m25 + 2*m35 + m45);
    end
    
    % Loop for Backward Sweep (Calculation of Co-states)
    for i = 1:M
        j = M + 2 - i;

        % Calculate right-hand side of co-state equations using backward difference and current state values
        m11 = -B*a(j) + lambda1(j)*((1-a(j))*b*I(j)/no(j) + (a(j)*0.103 + u)) - lambda2(j)*(1-a(j))*b*I(j)/no(j) - lambda3(j)*0.103;
        m12 = (lambda1(j) - lambda2(j))*(b*(1-a(j)*S(j)/no(j))) + lambda2(j)*(delt + u + gam) - lambda4(j)*gam - lambda5(j)*delt;
        m13 = lambda3(j)*(the + tel + u) - lambda4(j)*the - lambda5(j)*tel;
        m14 = -A + lambda4(j)*(gam + u + u3) - lambda5(j)*gam;
        m15 = -lambda1(j)*e + lambda5(j)*(e+u);

        m21 = -B*0.5*(a(j) + a(j-1)) + (lambda1(j) - h2*m11) *((1-(0.5*(a(j) + a(j-1))))*b*0.5*(I(j) + I(j-1))/no(j) + ((0.5*(a(j) + a(j-1)))*0.103 + u)) - (lambda2(j) - h2*m12)*(1-(0.5*(a(j) + a(j-1))))*b*0.5*(I(j) + I(j-1))/no(j) - (lambda3(j) - h2*m13)*0.103;
        m22 = ((lambda1(j) - h2*m11) - (lambda2(j) - h2*m12))*(b*(1-(0.5*(a(j) + a(j-1))))*0.5*(S(j) + S(j-1))/no(j)) + (lambda2(j) - h2*m12)*(delt + u + gam) - (lambda4(j) - h2*m14)*gam - (lambda5(j) - h2*m15)*delt;
        m23 = (lambda3(j) - h2*m13)*(the + tel + u) - (lambda4(j) - h2*m14)*the - (lambda5(j) - h2*m15)*tel;
        m24 = -A + (lambda4(j) - h2*m14)*(gam + u + u3) - (lambda5(j) - h2*m15)*gam;
        m25 = -(lambda1(j) - h2*m11)*e + (lambda5(j) - h2*m15)*(e+u);

        m31 = -B*0.5*(a(j) + a(j-1)) + (lambda1(j) - h2*m21) *((1-(0.5*(a(j) + a(j-1))))*b*0.5*(I(j) + I(j-1))/no(j) + ((0.5*(a(j) + a(j-1)))*0.103 + u)) - (lambda2(j) - h2*m22)*(1-(0.5*(a(j) + a(j-1))))*b*0.5*(I(j) + I(j-1))/no(j) - (lambda3(j) - h2*m23)*0.103;
        m32 = ((lambda1(j) - h2*m21) - (lambda2(j) - h2*m22))*(b*(1-(0.5*(a(j) + a(j-1))))*0.5*(S(j) + S(j-1))/no(j)) + (lambda2(j) - h2*m22)*(delt + u + gam) - (lambda4(j) - h2*m24)*gam - (lambda5(j) - h2*m25)*delt;
        m33 = (lambda3(j) - h2*m23)*(the + tel + u) - (lambda4(j) - h2*m24)*the - (lambda5(j) - h2*25)*tel;
        m34 = -A + (lambda4(j) - h2*m24)*(gam + u + u3) - (lambda5(j) - h2*m25)*gam;
        m35 = -(lambda1(j) - h2*m21)*e + (lambda5(j) - h2*m25)*(e+u);

        m41 = -B*a(j-1) + (lambda1(j) - h2*m31) *((1-a(j-1))*b*0.5*(I(j) + I(j-1))/no(j) + (a(j-1)*0.103 + u)) - (lambda2(j) - h2*m32)*(1-a(j-1))*b*0.5*(I(j) + I(j-1))/no(j) - (lambda3(j) - h2*m33)*0.103;
        m42 = ((lambda1(j) - h2*m31) - (lambda2(j) - h2*m22))*(b*(1-a(j-1)))*0.5*(S(j) + S(j-1))/no(j) + (lambda2(j) - h2*m32)*(delt + u + gam) - (lambda4(j) - h2*m34)*gam - (lambda5(j) - h2*m35)*delt;
        m43 = (lambda3(j) - h2*m33)*(the + tel + u) - (lambda4(j) - h2*m34)*the - (lambda5(j) - h2*35)*tel;
        m44 = -A + (lambda4(j) - h2*m34)*(gam + u + u3) - (lambda5(j) - h2*m35)*gam;
        m45 = -(lambda1(j) - h2*m31)*e + (lambda5(j) - h2*m35)*(e+u);

        % Calculate the co-states using a 4th order Runge-Kutta method
        lambda1(j-1) = lambda1(j) - (h/6)*(m11 + 2*m21 + 2*m31 + m41);
        lambda2(j-1) = lambda2(j) - (h/6)*(m12 + 2*m22 + 2*m32 + m42);
        lambda3(j-1) = lambda3(j) - (h/6)*(m13 + 2*m23 + 2*m33 + m43);
        lambda4(j-1) = lambda4(j) - (h/6)*(m14 + 2*m24 + 2*m34 + m44);
        lambda5(j-1) = lambda5(j) - (h/6)*(m15 + 2*m25 + 2*m35 + m45);
    end
    
    % Define the function for the control variable
    temp=(((b*S.*I.*(lambda2 - lambda1))./no) + 0.103*S.*(lambda1 - lambda3))./(2*B*S);

    % Limiting the possible Screening rate to be between 0 to 90 percent
    a1 = min(0.9,max(0,temp));
    a = 0.5*(a1 + olda);

    temp1 = delta*sum(abs(a)) - sum(abs(olda - a));
    temp2 = delta*sum(abs(S)) - sum(abs(oldS - S));
    temp3 = delta*sum(abs(I)) - sum(abs(oldI - I));
    temp4 = delta*sum(abs(H)) - sum(abs(oldH - H));
    temp5 = delta*sum(abs(C)) - sum(abs(oldC - C));
    temp6 = delta*sum(abs(R)) - sum(abs(oldR - R));
    temp7 = delta*sum(abs(lambda1)) - sum(abs(oldlambda1 - lambda1));
    temp8 = delta*sum(abs(lambda2)) - sum(abs(oldlambda2 - lambda2));
    temp9 = delta*sum(abs(lambda3)) - sum(abs(oldlambda3 - lambda3));
    temp10 = delta*sum(abs(lambda4)) - sum(abs(oldlambda4 - lambda4));
    temp11 = delta*sum(abs(lambda5)) - sum(abs(oldlambda5 - lambda5));

    % Calculate the convergence criteria
    test = min(temp1, min(temp2, min(temp3, min(temp4, min(temp5, min(temp6, min(temp7, min(temp8, min(temp9, min(temp10,temp11))))))))));
end
 y(1,:) = t;
 y(2,:) = S;
 y(3,:) = I;
 y(4,:) = H;
 y(5,:) = C;
 y(6, :) = R;
 y(7, :) = a;


figure;
subplot(2, 3, 1);
plot(y(2,:));
title('Susceptible');
xlabel('t');
ylabel('S');

subplot(2, 3, 2);
plot(y(3,:) + y(4,:));
title('Infected');
xlabel('t');
ylabel('I');

subplot(2, 3, 3);
plot(y(5,:));
title('Cancer');
xlabel('t');
ylabel('C');

subplot(2, 3, 4);
plot(y(6,:));
title('Recovered');
xlabel('t');
ylabel('R');

subplot(2, 3, 5);
plot(y(7,:));
title('Screening Rate');
xlabel('t');
ylabel('a');

end
