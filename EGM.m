%% Solve the Consumption Savings Model using an Endogenous Grid Method (EGM)

clear
clc
close all
%% 1. Define parameters

% Numerical parameters
mpar.nk   = 100;   % Number of points on the asset grid
mpar.nz   = 2;    % Number of points on the log-productivity grid
mpar.crit = 1e-5; % Numerical precision
mpar.maxk = 6;    % Maximimum assets
mpar.mink = -9/4;    % Minimum Assets (equal to Borrowing Limit)
disp('Numerical parameters')
mpar % Display numerical parameters
% Economic Parameters
par.r     = 4/90;% Real Rate, the rate we receive on our savings
par.gamma = 1;    % Coeffcient of relative risk aversion
par.beta  = 0.95; % Discount factor
par.b     = mpar.mink; % Borrowing Limit
disp('Economic parameters')
par % Display economic parameters

%% 2. Generate grids, Meshes and Income
gri.k   = exp(linspace(log(1),log(mpar.maxk-mpar.mink+1),mpar.nk))-1+mpar.mink; %Define asset grid on log-linearspaced
prob.z  = [3/5, 2/5; 4/90,  86/90]; %This is the transition matrix, whether in a state
                                %of low/high income.
gri.z   = [1/9, 10/9]; %state of the income shock 
% Meshes of capital and productivity
[meshes.k,  meshes.z] = ndgrid(gri.k,gri.z);
Y = meshes.z+(1+par.r)*meshes.k; % Cash at hand (Labor income plus assets with dividend)

%% 3. Define utility functions

if par.gamma ==1
    util     = @(c)log(c); % Utility
    mutil    = @(c)1./(c);  % Marginal utility
    invmutil = @(c)inv(mutil);% inverse marginal utility, need for consumption
else
    util  = @(c)1/(1-par.gamma).*c.^(1-par.gamma); % Utility
    mutil = @(c)1./(c.^par.gamma); % Marginal Utility
    invmutil = @(c)inv(mutil); % inverse marginal utility
end

%% 4. Value Function Iteration

tic % Start timer
V    = zeros(mpar.nk,mpar.nz); % Initialize Value Function
distVF = 1; % Initialize Distance
iterVF = 1; % Initialize Iteration count
while distVF(iterVF)>mpar.crit % Value Function iteration loop: until distance is smaller than crit.
    % Update Value Function using Spline (thus off-grid search)
    [Vnew,kprime] = VFI_update_spline(V,Y,util,par,mpar,gri,prob); % Optimize given cont' value
    dd            = max(abs(Vnew(:)-V(:))); % Calculate distance between old guess and update

    V             = Vnew; % Update Value Function
    iterVF        = iterVF+1; %Count iterations
    distVF(iterVF)= dd;   % Save distance
end
time(1)=toc; % Save Time used for VFI
%% 5. Plot Policy Functions from VFI

figure(1)
plot(gri.k,kprime) % Plot policy functions
hold on
plot(gri.k,gri.k,'k--') % Add 45?? line
title('Policy Function from VFI') % Title and legend of the graph
legend({'low productivity','medium productivity','high productivity'},'Location','northwest')

%% 6. Endogenous Grid method using linear interpolation
% $$\frac{\partial u}{\partial c}\left[C^*(k',z)\right]=(1+r) \beta E_{z}\left\{\frac{\partialu}{\partial
% c}\left[C(k',z')\right]\right\}$$

tic % Reset timer
C     = (par.r.*meshes.k+meshes.z); 
%Initial guess for consumption policy: Available resources are:
% Y = meshes.z+(1+par.r)*meshes.k; % Cash at hand (Labor income plus assets with dividend)
% Expanded, this is: (z + par.r*k) + k.
% Thus our policy guess is to "roll over assets". We consume (z + par.r*k)
% and leave the "+ k" for the next period.
Cold  = C; % Save old policy
distEG  = 1; % Initialize Distance
iterEG  = 1; % Initialize Iteration count
while distEG>mpar.crit
    C      = EGM(Cold,mutil,invmutil,par,mpar,prob.z,meshes,gri); % Update consumption policy by EGM
    dd     = max(C(:)-Cold(:)); % Calculate Distance

    Cold   = C; % Replace old policy
    iterEG = iterEG+1; %count iterations
    distEG(iterEG) = dd;
end
[C,Kprimestar] = EGM(C,mutil,invmutil,par,mpar,prob.z,meshes,gri); %The first entry of kprimestar
%is to consume at the borrowing constraint. 
time(2)        = toc; %Time to solve using EGM
%% 7. Plot Policy Functions from Collocation and compare to VFI

figure(2) %Plot Policy Functions from Collocation
plot(gri.k,Kprimestar) % Plot Policy function from Collocation
hold on
plot(gri.k,gri.k,'k--') % Add 45?? line
title('Policy Function from EGM') % Title and Legend
legend({'low productivity','medium productivity','high productivity'},'Location','northwest')

figure(3) %Plot Differences in Policies
plot(gri.k,Kprimestar - kprime)
title('Difference in Policy Function')

%% 8. Compare times of algorithms
disp('Time to solve (VFI, EGM)')
disp(time)
disp('Iterations to solve (VFI, EGM)')
disp([iterVF iterEG])

%% FUNCTIONS (need to be copied to extra files or run as a "Live Script")

%% VF Update
function [Vnew,kprime] = VFI_update_spline(V,Y,util,par,mpar,gri,prob)
  % VFI_update_spline updates the value function (one VFI iteration) for the
  %  consumption-savings problem.
  % V (dimensions: k x z) is the old value function guess.
  % Y (dimensions: k x z) is a matrix of cash at hand UTIL is the felicity function.
  % PAR and MPAR are structures containing economic and numerical  parameters.
  % PROB (dimensions: z x z') is the transition probability matrix.

V      = reshape(V,[mpar.nk,mpar.nz]); % make sure that V has the right format dim1: k, dim2:z
kprime = zeros(size(V)); % allocate policy matrix
Vnew   = zeros(size(V)); % allocate new value matrix
EV     = par.beta.*V*prob.z';   % Calculate expected continuation value

for zz=1:mpar.nz % loop over Incomes
    ev_int = griddedInterpolant({gri.k},EV(:,zz),'spline'); % Prepare interpolant
    for kk=1:mpar.nk % loop of Assets
        f             = @(k)(-util(Y(kk,zz)-k)-ev_int(k)); % Define function to be minimized
        [kp,v]        = fminbnd(f,par.b,Y(kk,zz)); % Find minimum of f for savings between par.b and Y(kk,zz)
        Vnew(kk,zz)   =-v;  % Save value
        kprime(kk,zz) = kp; % Save policy
    end
end
Vnew=Vnew(:);
end

%% Policy update by EGM
function [C,Kprime] = EGM(C,mutil,invmutil,par,mpar,P,meshes,gri)
    %% This function iterates forward the consumption policies for the consumption Savings
    % model using the EGM method. C (k x z) is the consumption policy guess. MUTIL and INVMUTIL are
    % the marginal utility functions and its inverse. PAR and MPAR are parameter structures.
    % P is the transition probability matrix. MESHES and GRI are meshes and grids for income
    % (z) and assets (k).
%We start with an initial guess for the consumption policy above called
%"Cold". We use Cold as the argument "C". This is the first Cstar. As long
%as its positive, monotonically increasing, the guess suffices. Now we find
%a Kstar, which when given consumption which we just generated and the stochastic income
%,the budget constraint resolves. So now we have the values for Kstar on
%the exogenous grid points of k_t+1. We want to find all Kstar off-grid, so to find
%the in between values, we use the gridded interpolant, where our inputs
%are the grid points k_t+1. Thus, just like forward induction, the state
%tomorrow was the result of rational actions today. 
    mu     = mutil; % Calculate marginal utility from c'
    emu    = (1+par.r)*par.beta*1./(C);     % Calculate expected marginal utility
    Cstar  = C %meshes.z.*meshes.k.^(par.alpha)-meshes.k;     % Calculate cstar(m',z)
    Kstar  = (meshes.k-meshes.z+Cstar)/(1+par.r) 
    % recall that consumption = what's inside the utiity function, that is:
    % consumption = (1+r)k - k' + z
    % Solving for k (i.e., k endogenous):
    % k* = (k' - z + Cstar)/(1+r), which is what we have above for Kstar 
    Kprime = meshes.k; % initialze exogenous Capital Policy

    for z=1:mpar.nz % For massive problems, this can be done in parallel
        % generate savings function k(z,kstar(k',z))=k'
        Savings     = griddedInterpolant(gri.k,Kstar(:,z), 'linear');
        Kprime(:,z) = Savings(gri.k);     % Obtain k'(z,k) by interpolation

    end
    BC         = meshes.k<repmat(Kstar(1,:),mpar.nk,1); % Check Borrowing Constraint, for the
    %first row and only one state since it is in the loop. Households
    %cannot have assets lower than the first kstar. That implies that they
    %are consuming more. thus, they must go to the borrowing constraint to
    %increase their asset levels to match the kstar. 
    % Replace Savings for HH saving at BC
    Kprime(BC) = par.b; % Households with the BC flag choose borrowing contraint
    % generate consumption function c(z,k^*(z,k'))
    C          =(1+par.r)*meshes.k+meshes.z-par.b ; %Consumption update from budget constraint
    

end
