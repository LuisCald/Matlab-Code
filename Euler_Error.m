%% Solve the Stochastic Growth Model using a Collocation method

clear
clc
close all
%% 1. Define parameters

%% Define Numerical Parameters
mpar.nk   = 10;  % number of points on the capital grid
mpar.nz   = 2;    % number of points on the productivity grid
mpar.mink = 0.2;  % lowest point on the capital grid
mpar.maxk = 0.25;    % highest point on the capital grid
mpar.crit = 1e-6; % Precision up to which to solve the value function

%% Define Economic parameters
par.beta  = 0.95; % Discount factor
par.alpha = 0.5;    % Curvature of production function
par.gamma = 1;    % Coefficient of relative risk aversion
par.delta = 1;    % Depreciation
prob.z     = [0.875, 0.125; 0.125, 0.875];% Transition probabilities for productivity

%% Produce grids

gri.k = exp(linspace(log(mpar.mink),log(mpar.maxk),mpar.nk));
gri.z = [0.99,1.01];
%% Display Model
TablePar={'Discount Factor:', par.beta; 'Returns to Scale', par.alpha; ...
    'Relative Risk Aversion', par.gamma; 'Depreciation', par.delta};

TablePar
%% 2. Generate grids, Meshes and Income
% Meshes of capital and productivity
[meshes.k,  meshes.z]= ndgrid(gri.k,gri.z);
Y = meshes.z.*meshes.k.^par.alpha; % Calculate Output  $$Y=zK^{\alpha}$$
R = par.alpha*meshes.z.*meshes.k.^(par.alpha-1); % Calculate Rental Rate $$z\alpha /K^{1-\alpha}$$

%% 3. Define utility functions / marginal utility
if par.gamma ==1
    util  = @(c)log(c);
    mutil = @(c) 1./c;
else
    util  = @(c) 1./(1-par.gamma).*c.^(1-par.gamma);
    mutil = @(c) 1./(c.^par.gamma);
end

%% 4. Value Function Iteration

tic % Start timer
V      = zeros(mpar.nk,mpar.nz); % Initialize Value Function
dist   = 1; % Initialize Distance
count  = 1; % Initialize Iteration count
while dist(count)>mpar.crit % Value Function iteration loop: until distance is smaller than crit.
    % Update Value Function using off-grid search
    [Vnew,kprime] = VFI_update_interp(V,Y,util,par,mpar,gri,prob); % Update the Value Function
    dd            = max(abs(Vnew(:)-V(:))); % Calculate distance using the sup norm.
    V             = Vnew; % Replace Value Function
    count         = count+1; %Count iterations
    dist(count)   = dd;% Store distance between Value Functions
end
time(1)=toc; % Save Time used for VFI

%% 5. Plot Policy Functions from VFI

figure(1)
plot(gri.k,kprime) % Plot policy functions
hold on
plot(gri.k,gri.k,'k--') % Add 45?? line
title('Policy Function from VFI') % Title and legend of the graph
legend({'low productivity','medium productivity','high productivity'},'Location','northwest')

%% 6. Collocation method using splines
% Solve for a root of all Euler Equation at nodes (k,z)

% $$EulerError(k,z)=\frac{\partial u}{\partial c}\left[Y-K^*(k,z)\right]-
% \beta \alpha E\left\{zk^{\alpha-1}\frac{\partial u}{\partial c}\left[Y'-K^*(K^*(k,z),z')\right]\right\}$$

tic % Reset timer
Euler = @(K)(EulerError(K,Y,R,mutil,par,mpar,gri,prob,meshes)); % Define Euler Error Function inline
Kprimestar = fsolve(Euler,Y/2);% Find those parameters (Values at spline nodes) that set the error to zero at nodes.
Kprimestar = reshape(Kprimestar,[mpar.nk,mpar.nz]); %Reshape the policy function
time(2)=toc; %Time to solve using Collocation

%% 7. Plot Policy Functions from Collocation and compare to VFI
figure(2) %Plot Policy Functions from Collocation
plot(gri.k,Kprimestar) % Plot Policy function from Collocation
hold on
plot(gri.k,gri.k,'k--') % Add 45?? line
title('Policy Function from Collocation') % Title and Legend
legend({'low productivity','medium productivity','high productivity'},'Location','northwest')

figure(3) %Plot Differences in Policies
plot(gri.k,Kprimestar - kprime)
title('Difference in Policy Function')

%% 8. Compare times of algorithms
disp('Time to solve (VFI, Collocation)')
disp(time)

%% FUNCTIONS (need to be copied to extra files or run as a "Live Script")

%% Euler Error

function [EulerE] = EulerError(kprime,Y,R,mutil,par,mpar,gri,prob,meshes)
    % EulerError calculates the error from the Euler Equation for the stocahstic
    % growth model given a savings policy KPRIME (dimensions: k x z).
    % Y (dimensions: k x z) is income for all combinations of assets (k)
    % and productivity (z) on grid GRI.k and GRI.z respectively.
    % PROB is the Transition probability matrix (dimensions: z x z')
    % and MESHES the meshes of gri.z and gri.k.

kprime=reshape(kprime,[mpar.nk,mpar.nz]); % make sure that kprime comes as a matrix
if min(Y(:)-kprime(:))>0 && min(kprime(:))>0 % feasible policy. Min is only evaluated if max is true.
%Consumption and kprime must be positive
    MU  = mutil(Y-kprime); %Calculate marginal utility of consumption (y(k,z)-k'(k,z))
    % Calculate expected marginal utility E (mu(k',z')) (dimensions k' x z)
    EMU = par.beta*(R.*MU)*prob.z';%since R+1-par.delta is is just R
    % use this to define an interpolant
    EMUfun  = griddedInterpolant({gri.k,gri.z},EMU,'linear'); %generates the input and output of this 
%interpolated function 
    EMU_val = EMUfun(kprime,meshes.z); % Evaluate interpolant at k'(k,z) to obtain f(k,z) = E_z mu(k'(k,z),z')
    %F = MU-EMU_val
    %nodes = cos((2*(0:mpar.nk-1)+1)/(2*mpar.nk)*pi);
    %EE = inner(F,dirac(gri.k-nodes)) ; % Write up Euler Equation Error
    EE = MU-EMU_val ; 
    EulerE=EE(:); % Stack as vector
else % infeasible policy
    EulerE=ones(size(Y(:)))*1e15;
end
end

%% Value Function Update
function [Vnew,kprime] = VFI_update_interp(V,Y,util,par,mpar,gri,prob)
V=reshape(V,[mpar.nk,mpar.nz]);
kprime = zeros(size(V));
Vnew   = zeros(size(V));
EV     =par.beta* V* prob.z';   % Calculate expected continuation value

for zz=1:mpar.nz
    ev_int= griddedInterpolant({gri.k},EV(:,zz),'spline');
    for kk=1:mpar.nk
        f             = @(k)(-util(Y(kk,zz)-k)-ev_int(k));
        [kp,v]        = fminbnd(f,0,Y(kk,zz));
        Vnew(kk,zz)   = -v;
        kprime(kk,zz) = kp;
    end
end
Vnew=Vnew(:);
end

