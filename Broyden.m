%% Broyden Method

%% 0. Clean up

clear
clc
close all

%% Define Numerical parameters
mpar.nk   = 10; % number of points on the capital grid
mpar.nz   = 2;   % number of points on the productivity grid
mpar.mink = .1; % lowest point on the capital grid
mpar.maxk = .4;   % highest point on the capital grid
mpar.crit = 1e-6;% Precision up to which to solve the VF
%% Define Economic parameters
par.beta = 0.95; % Discount factor
par.alpha = 0.5;   % Curvature of production function
par.gamma = 1;   % Coefficient of relative risk aversion
par.delta = 1;   % Depreciation rate
par.MPK  = ((1-par.beta)/par.beta+par.delta); % Marginal product of capital in steady state

%par.MPK  = ((1-par.beta)/par.beta+par.delta); % Marginal product of capital in steady state
prob.z = [0.875 0.125; 0.875 .125];      % Transition probability for productivity

%% Produce Grids
grid.k = exp(linspace(log(mpar.mink),log(mpar.maxk),mpar.nk)); % Grid for capital
grid.z =[.9 1.1]; %grid for productivity

%% Define felicity functions

if par.gamma==1
    util  = @(c)log(c); % felicity function (Vectorized)
    mutil = @(c)1./c;   % marginal felicty
else
    util  = @(c)c.^(1-par.gamma)./(1-par.gamma); % felicity function (Vectorized)
    mutil = @(c)1./(c.^par.gamma);   % marginal felicty
end


%% Calculate Consumption and Utility for Capital Choices
[meshes.k,  meshes.kprime, meshes.z]= ndgrid(grid.k,grid.k,grid.z);
Y = meshes.z.*meshes.k.^(par.alpha)+ (1-par.delta).*meshes.k; %Income/Resources
C = meshes.z.*meshes.k.^(par.alpha)-meshes.kprime ; %Consumption

U      = log(C) ; %Dimensions k x k' x z
U(C<0) = -Inf; % Disallow negative consumption

%% Value Funtion Iteration (Spline, off-grid)
V     = zeros(mpar.nk, mpar.nz); % Initialize the value function 
dist  = 9999; % term to store the distance of value functions between iterations 
count = 1;
tic
%We interpolate the value function and use the interpolation for each
%iteration. 
while dist(count)>mpar.crit
   count = count+1; %count the number of iterations
   Vnew = VF_spline(V, util, par, mpar,grid,prob,Y);
   dist(count) = max(abs(Vnew(:)-V(:))); % distance between updates
   V = (Vnew); % Remove superfluous second dimension
end
toc
[~,kprime] = VF_spline(V, util, par, mpar,grid,prob,Y); %create policy functions (in indexes)

%% Do the same as above but with a solver
V     = zeros(mpar.nk, mpar.nz); % Initialize the value function 
V     = V(:);
dist_V = @(V)(V-VF_spline(V, util, par, mpar,grid,prob,Y));
tic
[V,fval,iter, distBroyden]=broyden(dist_V,V, mpar.crit, mpar.crit, 500);
toc
%% Plot results
figure(1)
semilogy(dist(2:end))
hold on
semilogy(distBroyden(2:end))
title('Distance between iterations')

figure(2)
plot(grid.k,kprime)
hold on
share_saved = par.beta*par.alpha;
plot(grid.k,share_saved*(grid.k'.^par.alpha*grid.z),'--')

plot(grid.k,grid.k,'k:')
legend({'Capital policy (numerical, low prod)', 'Capital policy (numerical, high prod)',...
    'Capital policy (analytical, low prod)','Capital policy (analytical, high prod)', '45 degree line'})
plot([par.MPK.^(1/(par.alpha-1))* (par.alpha*grid.z(1)).^(1/(1-par.alpha)) par.MPK.^(1/(par.alpha-1))* (par.alpha*grid.z(1)).^(1/(1-par.alpha))],[grid.k(1) grid.k(end)],'k')
plot([par.MPK.^(1/(par.alpha-1))* (par.alpha*grid.z(end)).^(1/(1-par.alpha)) par.MPK.^(1/(par.alpha-1))* (par.alpha*grid.z(end)).^(1/(1-par.alpha))],[grid.k(1) grid.k(end)],'k')
%% AUXILIARY FUNCTIONS
function [Vnew, policy] = VF_spline(V, util, par, mpar,grid,prob,Y)
% This function does one update step in value-function iteration
% it assumes the value function is represented by a spline with node
% values V. Maximization is off-grid.
V      = reshape(V,[mpar.nk, mpar.nz]); % V is a matrix
policy = nan(size(V)); % container for policy
Vnew   = nan(size(V)); % container for VF update
EV     = par.beta*V* prob.z'; % Calculate expected value
ev_int = griddedInterpolant({grid.k, grid.z},EV,'spline'); % generate an interpolant
for zz =1:mpar.nz %loop over productivity
    kp = 0;
    for kk =1:mpar.nk %loop over current capital
        y = Y(kk,zz);%gri.z(zz).*gri.k(kk).^par.alpha +(1-par.delta)*gri.k; % define resources
        f = @(k)(-(util(y-k) + ev_int({k,grid.z(zz)}))); % objective function t.b. maximized w.r.t. k
        [kp, v] = fminbnd(f,0, y); % mimimization of -f
        Vnew(kk,zz) = -v;
        policy(kk,zz) = kp;
    end
end
Vnew=Vnew(:);
end

function [xstar, fval, iter, distF] =broyden(f,x0,critF, critX, maxiter)
    %The function here uses the "good" broyden algorithm to solve for a
    %root of function f
    % X0 is the starting guess, CRITX and CRITF  are the precisions for x and f
    % MAXITER is the maximum number of iterations
    
    distF = NaN(1,maxiter); 
    distF(1)=9999;
    distX = 9999;
    iter =1; %count the number of iterations
    xnow = x0(:); % x needs to be a column
    Fnow = f(xnow); % current function value
    Fnow = Fnow(:); % needs to be a column
    Bnow = eye(length(xnow)); % initial guess for inverse Jacobian
    
    while distF(iter)>critF && distX>critX && iter<maxiter
        iter = iter+1; % count iterations
        Fold = Fnow; % store function value
        xold = xnow; % store old argument
        xnow = xold - Bnow*Fold; % upadate for root guess, I changed to Fold since I think it is correct
        Fnow = f(xnow); % update function value
        Dx = xnow - xold; % Change in x
        DF = Fnow- Fold; %Change in F
        % update the inverse Jacobian
        Bnow = Bnow + (Dx - Bnow*DF)*(Dx'*Bnow)/(Dx'*Bnow*DF);
        distF(iter) = max(abs(Fnow)); %how far away from root
        distX = max(abs(Dx)); % how much did x change
    end
    fval = Fnow; xstar=xnow;
    if maxiter == iter
        warning('it did not converge !!!!')
    end
end