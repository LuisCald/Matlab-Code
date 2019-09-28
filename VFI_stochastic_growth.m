%% Solve the Cake eating problem

clc
clear
close all

%% Define Numerical Parameters
mpar.nk   = 100;  % number of points on the capital grid
mpar.nz   = 2;    % number of points on the productivity grid
mpar.mink = 0.1;  % lowest point on the capital grid
mpar.maxk = 0.4;    % highest point on the capital grid
mpar.crit = 1e-6; % Precision up to which to solve the value function

%% Define Economic parameters
par.beta  = 0.95; % Discount factor
par.alpha = 0.5;    % Curvature of production function
par.gamma = 1;    % Coefficient of relative risk aversion
par.delta = 1;    % Depreciation
prob.z     = [0.875, 0.125; 0.125, 0.875];% Transition probabilities for productivity

%% Produce grids

grid.k = exp(linspace(log(mpar.mink),log(mpar.maxk),mpar.nk)); %1x100. The logarithm is taken sometimes 
%since the curvature of the value function and the policy function is
%greater for the lower grid points. The log makes the grid a bit finer to
%catch these movements. 
grid.z = [0.9,1.1]; %1x2
%% Display Model
TablePar={'Discount Factor:', par.beta; 'Returns to Scale', par.alpha; ...
    'Relative Risk Aversion', par.gamma; 'Depreciation', par.delta};

TablePar
%% Define utility functions

if par.gamma ==1
    util  = @(c)log(c);
    mutil = @(c) 1./c;
else
    util  = @(c) 1/(1-par.gamma).*c.^(1-par.gamma);
    mutil = @(c) 1./(c.^par.gamma);
end

%% Calculate Consumption and Utility for Capital Choices
[meshes.k,  meshes.kprime, meshes.z]= ndgrid(grid.k,grid.k,grid.z); %ndgrid creates a rectangular grid. The grid is a 100x100x2. 
                                                                            %and it returns 3 of them 
                                                                    
Y = meshes.z.*meshes.k.^par.alpha + (1-par.delta).*meshes.k;
C = Y-meshes.kprime;

U      = util(C); %Dimensions k x k' x z
U(C<0) = -Inf; % Disallow negative consumption

%% Value Function Iteration
%So here, we have a set of grid points. Each iteration runs through the
%grid. We find the argmax of the value function, k'1. Then, we iterate
%again, with a new grid point as the starting point to find the argmax for this one, k'2.  
%In the end, we find a policy function vector h(Ki)= K' for each starting point (each grid point)
%we will have a vector of kprimes, which is the
%size of the number of grid points. The idea is for each grid point we compute
%the value function, find the k' that maximizes the value function [~,policy] and set
%that equal to the "i"th element in the k' vector. The point is after
%finding the k' that maximizes the bellman (given my guess), I evaluate the
%bellman and see how it compares to my initial guess (V=0). If the distance
%is still larger than the critical value, we iterate again with the new
%value function as our "guess". We continue doing this until the distance
%is closer and closer. Convergence is LINEAR! c*(1-beta)^(n), where c is
%the critical value 
%
tic 
V     = zeros(mpar.nk,mpar.nz); % Our guess is zero, this implies that the initial EV is zero
%The next period, we know K' is zero as well, thus, the first value
%function is just out utility function U(C) with K'=0
dist  = 9999;
count = 1;
while dist(count)>mpar.crit
    count       = count+1;                % count the number of iterations
    EV          = par.beta* V* prob.z';   % Calculate expected continuation value
    EVfull      = repmat(reshape(EV,[1 mpar.nk mpar.nz]),[mpar.nk 1 1]); % Copy Value to second dimension
    %This is repmat(a,b), where b in this case is [100 1 1]. So we multiply
    %the newly reshaped EV(1 100 2) by [100 1 1] = [100 100 2]. In the
    %notation in the powerpoint, this multiplication process is represented
    %by the "i" vector. 

    Vnew        = max(U + EVfull,[],2);   % Update Value Function, taking max of 2nd dimension
    dist(count) = max(abs(Vnew(:)-V(:))); % Calculate distance between old guess and update
    V           = squeeze(Vnew);          % Copy update to value function. Squeeze removes the 
                                            %dimensions of length 1.
                                            %Compare Vnew to V to see
end
toc
%% Produce Policy functions
[~,policy]  = max(U + EVfull,[],2); 
kprime      = grid.k(squeeze(policy));%it produced the policy function for each possible starting point
%and then repeated for the productivity state. 

%% Plots
figure(1)
semilogy((dist(2:end)))
title('Distance between two updates of V -logscale')

figure(2)
plot(grid.k,kprime(:,1))
hold on
plot(grid.k,kprime(:,2))
hold on
plot(grid.k,grid.k,'k--')
legend({'Capital policy Low','Capital Policy High','45 degree line'})
title('Policy functions')
%The intersection shows that a steady state exists
figure(3) 
plot(grid.k, Vnew(:,1))
hold on 
plot(grid.k,Vnew(:,2))
legend({'Low Productivity', 'High Productivity'})
title('Value functions')

