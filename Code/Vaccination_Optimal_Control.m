clear; clc; close all;

% Call the function 'vaccination_control' with parameters and store the output in 'y'
y = vaccination_control(100,0.8554, 0.556,0.556, 0.5, 0.014156, 0.108696, 0.057, 0.2);

function y = vaccination_control(T, Lambda, b,bv, delt, u,u3, gam, e)

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

% Define model parameters - Cost of Cancer treatment (A) and cost per vaccination (b)
A = 60000;
B = 2000;

% Initialize state and co-state variables
U=zeros(1,M+1);
I=zeros(1,M+1);
C=zeros(1,M+1);
R=zeros(1,M+1);
no=ones(1,M+1);

% Set initial conditions for state variables
x0 = [56.47,3.8465, 0.0128, 0];

U(1) = x0(1);
I(1)=x0(2);
C(1)=x0(3);
R(1)=x0(4);

lambda1=zeros(1,M+1);
lambda2=zeros(1,M+1);
lambda3=zeros(1,M+1);
lambda4=zeros(1,M+1);

v = zeros(1,M+1);

while(test < 0)
    
    % Store old values of states and co-states for convergence test
    oldv = v;
    oldU = U;
    oldI = I;
    oldC = C;
    oldR = R;

    oldlambda1 = lambda1;
    oldlambda2 = lambda2;
    oldlambda3 = lambda3;
    oldlambda4 = lambda4;
    
    % Loop for forward sweep (calculate states)
    for i = 1:M
        no(i) = U(i) + I(i) + C(i) + R(i);

        % Calculate right-hand side of state equations using forward difference and current co-state values
        m11 = (1-v(i))*Lambda + (e * R(i)) - b*U(i)*I(i)/no(i) - u*U(i);
        m12 = b*U(i)*I(i)/no(i) + bv*0.05*Lambda*v(i)*I(i)/no(i)  - (delt + u + gam) * I(i);
        m13 = gam*I(i) - (u + u3)*C(i);
        m14 = delt * I(i) - (e + u) * R(i);

        m21 = (1-v(i))*Lambda + (e * (R(i)+h2*m14)) - b*(U(i) + h2*m11)*(I(i) + h2*m12)/no(i) - u*(U(i) + h2*m11);
        m22 = b*(U(i) + h2*m11)*(I(i) + h2*m12)/no(i) + bv*0.05*Lambda*v(i)*(I(i) + h2*m12)/no(i)  - (delt + u + gam) * (I(i) + h2*m12);
        m23 = gam*(I(i) + h2*m12) - (u + u3)*(C(i) + h2*m13);
        m24 = delt * (I(i) + h2*m12) - (e + u) * (R(i)+h2*m14);

        m31 = (1-v(i))*Lambda + (e * (R(i)+h2*m24)) - b*(U(i) + h2*m21)*(I(i) + h2*m22)/no(i) - u*(U(i) + h2*m21);
        m32 = b*(U(i) + h2*m21)*(I(i) + h2*m22)/no(i) + bv*0.05*Lambda*v(i)*(I(i) + h2*m22)/no(i)  - (delt + u + gam) * (I(i) + h2*m22);
        m33 = gam*(I(i) + h2*m22) - (u + u3)*(C(i) + h2*m23);
        m34 = delt * (I(i) + h2*m22) - (e + u) * (R(i)+h2*m24);

        m41 = (1-v(i))*Lambda + (e * (R(i)+h2*m34)) - b*(U(i) + h2*m31)*(I(i) + h2*m32)/no(i) - u*(U(i) + h2*m31);
        m42 = b*(U(i) + h2*m31)*(I(i) + h2*m32)/no(i) + bv*0.05*Lambda*v(i)*(I(i) + h2*m32)/no(i)  - (delt + u + gam) * (I(i) + h2*m32);
        m43 = gam*(I(i) + h2*m32) - (u + u3)*(C(i) + h2*m33);
        m44 = delt * (I(i) + h2*m32) - (e + u) * (R(i)+h2*m34);
     
        % Update states using 4th order Runge-Kutta approximation
        U(i+1) = U(i) + (h/6)*(m11 + 2*m21 + 2*m31 + m41);
        I(i+1) = I(i) + (h/6)*(m12 + 2*m22 + 2*m32 + m42);
        C(i+1) = C(i) + (h/6)*(m13 + 2*m23 + 2*m33 + m43);
        R(i+1) = R(i) + (h/6)*(m14 + 2*m24 + 2*m34 + m44);
    end
    
    % Loop for Backward Sweep (Calculation of Co-states)
    for i = 1:M
        j = M + 2 - i;

        % Calculate right-hand side of co-state equations using backward difference and current state values
        m11 = -B*v(j) + lambda1(j)*(b*I(j)/no(j) + u) - lambda2(j)*(b*I(j)/no(j));
        m12 = (lambda1(j) - lambda2(j))*(b*U(j)/no(j)) + lambda2(j)*((delt + u + gam) - 0.05*bv*Lambda*v(i)/no(j)) - lambda3(j)*gam - lambda4(j)*delt;
        m13 = -A + lambda3(j)*(u + u3);
        m14 = -lambda1(j)*e + lambda4(j)*(e+u);

        m21 = -B*(v(j) + v(j-1)) + (lambda1(j) - h2*m11)*(b*(I(j) + I(j-1))/no(j) + u) - (lambda2(j) - h2*m12)*(b*(I(j) + I(j-1))/no(j));
        m22 = ((lambda1(j) - h2*m11) - (lambda2(j) - h2*m12))*(b*(U(j) + U(j-1))/no(j)) + (lambda2(j) - h2*m12)*((delt + u + gam) - 0.05*bv*Lambda*(v(j) + v(j-1))/no(j)) - (lambda3(j) - h2*m13)*gam - (lambda4(j) - h2*m14)*delt;
        m23 = -A + (lambda3(j) - h2*m13)*(u + u3);
        m24 = -(lambda1(j) - h2*m11)*e + (lambda4(j) - h2*m14)*(e+u);

        m31 = -B*(v(j) + v(j-1)) + (lambda1(j) - h2*m21)*(b*(I(j) + I(j-1))/no(j) + u) - (lambda2(j) - h2*m22)*(b*(I(j) + I(j-1))/no(j));
        m32 = ((lambda1(j) - h2*m21) - (lambda2(j) - h2*m22))*(b*(U(j) + U(j-1))/no(j)) + (lambda2(j) - h2*m22)*((delt + u + gam) - 0.05*bv*Lambda*(v(j) + v(j-1))/no(j)) - (lambda3(j) - h2*m23)*gam - (lambda4(j) - h2*m24)*delt;
        m33 = -A + (lambda3(j) - h2*m23)*(u + u3);
        m34 = -(lambda1(j) - h2*m21)*e + (lambda4(j) - h2*m24)*(e+u);

        m41 = -B*(v(j-1)) + (lambda1(j) - h2*m31)*(b*(I(j) + I(j-1))/no(j) + u) - (lambda2(j) - h2*m32)*(b*(I(j) + I(j-1))/no(j));
        m42 = ((lambda1(j) - h2*m31) - (lambda2(j) - h2*m32))*(b*(U(j) + U(j-1))/no(j)) + (lambda2(j) - h2*m32)*((delt + u + gam) - 0.05*bv*Lambda*(v(j-1))/no(j)) - (lambda3(j) - h2*m33)*gam - (lambda4(j) - h2*m34)*delt;
        m43 = -A + (lambda3(j) - h2*m33)*(u + u3);
        m44 = -(lambda1(j) - h2*m31)*e + (lambda4(j) - h2*m34)*(e+u);

        % Calculate the co-states using a 4th order Runge-Kutta method
        lambda1(j-1) = lambda1(j) - (h/6)*(m11 + 2*m21 + 2*m31 + m41);
        lambda2(j-1) = lambda2(j) - (h/6)*(m12 + 2*m22 + 2*m32 + m42);
        lambda3(j-1) = lambda3(j) - (h/6)*(m13 + 2*m23 + 2*m33 + m43);
        lambda4(j-1) = lambda4(j) - (h/6)*(m14 + 2*m24 + 2*m34 + m44);
    end

    % Define the function for the control variable
    temp=(0.5*no - 0.015*b*I)./(b*no);

    % Limiting the possible Vaccination rate to be between 0 to 100 percent
    v1 = min(1,max(0,temp));
    v = 0.5*(v1 + oldv);
    
    temp1 = delta*sum(abs(v)) - sum(abs(oldv - v));
    temp2 = delta*sum(abs(U)) - sum(abs(oldU - U));
    temp3 = delta*sum(abs(I)) - sum(abs(oldI - I));
    temp4 = delta*sum(abs(C)) - sum(abs(oldC - C));
    temp5 = delta*sum(abs(R)) - sum(abs(oldR - R));
    temp6 = delta*sum(abs(lambda1)) - sum(abs(oldlambda1 - lambda1));
    temp7 = delta*sum(abs(lambda2)) - sum(abs(oldlambda2 - lambda2));
    temp8 = delta*sum(abs(lambda3)) - sum(abs(oldlambda3 - lambda3));
    temp9 = delta*sum(abs(lambda4)) - sum(abs(oldlambda4 - lambda4));

    % Calculate the convergence criteria
    test = min(temp1, min(temp2, min(temp3, min(temp4, min(temp5, min(temp6, min(temp7, min(temp8, temp9))))))));
end
 y(1,:) = t;
 y(2,:) = U;
 y(3,:) = I;
 y(4,:) = C;
 y(5, :) = R;
 y(6, :) = v;


figure;
subplot(2, 3, 1);
plot(y(2,:));
title('Unvaccinated');
xlabel('t');
ylabel('U');

subplot(2, 3, 2);
plot(y(3,:) + y(4,:));
title('Infected');
xlabel('t');
ylabel('I');

subplot(2, 3, 3);
plot(y(4,:));
title('Cancer');
xlabel('t');
ylabel('C');

subplot(2, 3, 4);
plot(y(5,:));
title('Recovered');
xlabel('t');
ylabel('R');

subplot(2, 3, 5);
plot(y(6,:));
title('Vaccination Rate');
xlabel('t');
ylabel('v');

end
