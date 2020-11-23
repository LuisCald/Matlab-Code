%% Solution to a stochastic growth model 
% The code cosiders off-grid choices. The planning problem is solved 
% by value function iteration.

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


%% Display Model parameters
TablePar = {'Discount Factor:', par.beta; ...
            'Returns to Scale', par.alpha; ...
            'Relative Risk Aversion', par.gamma;...
            'Depreciation Rate', par.delta};

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
while dist(count)>mpar.crit
   count = count+1; %count the number of iterations
   Vnew = VF_spline(V, util, par, mpar,grid,prob,Y); % the function we define
   dist(count) = max(abs(Vnew(:)-V(:))); % distance between updates
   V = (Vnew); % Remove superfluous second dimension
end
toc
[~,kprime] = VF_spline(V, util, par, mpar,grid,prob,Y); %create policy functions (in indexes)


%% Plot results
figure(1)
semilogy(dist(2:end))
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
% This function does one update step in value-function iteration. 
% it assumes the value function is represented by a spline with node
% values V. Maximization is off-grid.
%Vnew are the value functions at the choice variable value which maximize
%the value function 
V      = reshape(V,[mpar.nk, mpar.nz]); % V is a matrix 10x2
policy = nan(size(V)); % container for policy, 10x2
Vnew   = nan(size(V)); % container for VF update, 10x2
EV     = par.beta*V* prob.z'; % Calculate expected value
ev_int = griddedInterpolant({grid.k, grid.z},EV,'spline'); % generate an interpolant. Arguments: over what grid, on what function, what method
for zz =1:mpar.nz %loop over productivity states as required. 
    % kp = 0;
    for kk =1:mpar.nk %loop over current capital
        y = Y(kk,zz);%gri.z(zz).*gri.k(kk).^par.alpha +(1-par.delta)*gri.k; % available resources
        f = @(k)(-(util(y-k) + ev_int({k,grid.z(zz)}))); % non-linear objective function t.b. maximized w.r.t. k
        [kp, v] = fminbnd(f,0, y); 
% mimimization of -f for k in the interval 0 and y (just means we could
% invest nothing, so bounded below by zero and we are bounded above by the
% available resources, y
        Vnew(kk,zz) = -v;
        policy(kk,zz) = kp; % kp for k-prime
    end
end
%So now for each starting capital point and each state z, we have a
%parametric evaluation for the valuation guess. 
Vnew=Vnew(:);
end