%% Solve a Bewley model of money by simulating one agent over time
clear
clc
close all
%% 1. Define parameters

mpar.nk   = 100;   % Number of points on the asset grid
mpar.nz   = 2;    % Number of points on the log-productivity grid
mpar.crit = 1e-5; % Numerical precision
mpar.maxk = 1;    % Maximimum assets
mpar.mink = 0;    % Minimum Assets (equal to Borrowing Limit)
mpar.T    = 100000;
disp('Numerical parameters')
mpar % Display numerical parameters
% Economic Parameters
par.r     = 0;% Real Rate
par.gamma = 1;    % Coeffcient of relative risk aversion
par.beta  = 0.95; % Discount factor
par.b     = mpar.mink; % Borrowing Limit
disp('Economic parameters')
par % Display economic parameters

%% 2. Generate grids, Meshes and Income
gri.k   = exp(linspace(0,log(mpar.maxk-mpar.mink+1),mpar.nk))-1+mpar.mink; %We define our grid in such a fancy way to be between 0 and 1
prob.z  = [3/5, 2/5; 4/90,  86/90];
gri.z   = [1/9, 10/9];
% Meshes of capital and productivity
[meshes.k,  meshes.z] = ndgrid(gri.k,gri.z);
Y = meshes.z + meshes.k*(1+par.r); % Cash at hand (Labor income plus assets cum dividend) for households

%% 3. Define utility functions

if par.gamma ==1
    util     = @(c)log(c); % Utility
    mutil    = @(c) 1./c;  % Marginal utility
    invmutil = @(mu) 1./mu;% inverse marginal utility
else
    util     = @(c) 1/(1-par.gamma).*c.^(1-par.gamma); % Utility
    mutil    = @(c) 1./(c.^par.gamma); % Marginal utility
    invmutil = @(mu) 1./(mu.^(1./par.gamma)); % inverse marginal utility
end

%% 4. Endogenous Grid method using linear interpolation
% $$\frac{\partial u}{\partial c}\left[C^*(k',z)\right]=(1+r) \beta E_{z}\left\{\frac{\partialu}{\partial
% c}\left[C(k',z')\right]\right\}$$

tic % Reset timer
C     = (meshes.z  + par.r*meshes.k); %Initial guess for consumption policy: roll over assets, can eat up everything since no borrowing limit
Cold  = C; % Save old policy
distEG  = 1; % Initialize Distance
iterEG  = 1; % Initialize Iteration count
while distEG>mpar.crit
    C      = EGM(Cold,mutil,invmutil,par,mpar,prob.z,meshes,gri); % Update consumption policy by EGM
    dd     = max(abs(C(:)-Cold(:))); % Calculate Distance

    Cold   = C; % Replace old policy
    iterEG = iterEG+1; %count iterations
    distEG(iterEG) = dd;
end
[C,kprime] = EGM(C,mutil,invmutil,par,mpar,prob.z,meshes,gri);
time(2)        = toc; %Time to solve using EGM
%% 5. Plot Policy Functions from Collocation and compare to VFI

figure(1) %Plot Policy Functions from Collocation
plot(gri.k,kprime) % Plot Policy function from Collocation
hold on
plot(gri.k,gri.k,'k--') % Add 45?? line
title('Policy Function from EGM') % Title and Legend
legend({'low productivity','medium productivity','high productivity'},'Location','northwest')


%% 6. Simulate the economy
%Here we begin the process of policy functions inducing a markov chain. We
%first interpolate over our grid of states and find our kprime, which is
%our policy function. 
% Define an off-grid savings function by interpolation
tic
Saving = griddedInterpolant({gri.k,gri.z},kprime); %Using the kprime from above, we interpolate. maps today to tomorrow
% Simulate the exogeneous state
PI = cumsum(prob.z,2); %used to compare the cdfs
epsilon = rand(1,mpar.T); %Random numbers for simulation
S  = randi(mpar.nz,1,mpar.T); % Starting value. This selects INTEGERS between 1 and 2 and puts them in a 1xT matrix
k  = zeros(1,mpar.T); %Starting value for NEXT PERIODS ASSETS assets
for t=2:mpar.T   %Simulates for the first state, X_1, second state X_2 and so forth 
    S(t) = max(min(sum(PI(S(t-1),:)<epsilon(t))+1,mpar.nz),1); %we take the max of this min or 1, safeguard against the zero probability states, 
%which means we go to state 2 or stay in state 1 
    k(t) = Saving({k(t-1),gri.z(S(t))}); %using our gridded interpolant, we evaluate savings given the state 
%generated above
end
toc
figure(2)
histogram(k(10001:end),'BinEdges',(gri.k(1:end-1) + gri.k(2:end))/2,'Normalization','probability')%Bin Edges creates the bounds of the bins
title('Distribution of asset holdings')
disp('Average Asset holdings')
disp(mean(k(10001:end)))
hold on

%% 7. Obtain asset holdings by defining a Markov chain with policies and grid
%This is the second step. We found our kprime from interpolation as stated in the powerpoint. 
%Now we obtain the ergodic distirbution os states the planning problem
%INDUCES WIHTOUT SIMULATION. Also, we ahve constructed it so that it operates
%SOLELY on the grid! We must
%then now define indices to formulate the weights. The weights constitute
%how much weight to apply to the value function at gri.k(idk+1) and
%gri.k(idk) at the same income state. 
tic
[~,idk]                 = histc(kprime,gri.k); % find the next smallest INDEX i* relative to s*. 
%The actual formula shows how many kprimes are in the bin gri.k. gri.k
%specifies the end points of the bin. 
idk(kprime<=gri.k(1))   = 1; % For those outside of the grid, just in case the policy is out of bounds, we keep it to remain in the index set
idk(kprime>=gri.k(end)) = mpar.nk-1; % remain in the index set
distance    = kprime - gri.k(idk); %defining the numerator of the weight, where gri.k(idk) is the smallest index
weightright = distance./(gri.k(idk+1)-gri.k(idk)); %This is exactly the linear interpolation weights
weightleft  = 1-weightright; 
%Creating our transition matrix with the weights. 
Trans_array = zeros(mpar.nk,mpar.nz,mpar.nk,mpar.nz); %Assets now, Income now, Assets next, Income next
for zz=1:mpar.nz % all current income states
    for kk=1:mpar.nk % all current asset states
        Trans_array(kk,zz,idk(kk,zz),:)   =  weightleft(kk,zz) *reshape(prob.z(zz,:),[1 1 1 mpar.nz]);
        Trans_array(kk,zz,idk(kk,zz)+1,:) =  weightright(kk,zz)*reshape(prob.z(zz,:),[1 1 1 mpar.nz]);
    end
end
%extrapolation can cause negative weight entries, so everything must be on
%the grid!
%Creating our policy induced gamma matrix
Gamma=reshape(Trans_array,[mpar.nk*mpar.nz, mpar.nk*mpar.nz]); %We get our transition matrix, Gamma 
%This is the left unit eigenvector the policy function induces
[x,d]=eigs(Gamma',1); % x_{t+1} = x_t P_{K} . "d" generates the top eigenvalues, where the second argument defines the top "1" for example here. 
                                                                            %Here we just ask for the greatest eigenvalue, which is equal to 1
x=x./sum(x); %"x" above is just the eigenvector. Here we divide everything by the sum, which is 1! Eigenvector is a direction, can be to any scalar. 
%There should be all positive. 
toc
figure(2)
marginal_k=sum(reshape(x,[mpar.nk, mpar.nz]),2); % marginal histogram of assets. Second dimension, reshaping x
bar(gri.k,marginal_k)
title('Distribution of asset holdings')
Average_K= sum(marginal_k'.*gri.k)%to compare moments 
legend('Simulation','Direct Calculation')

%% SUBFUNCTIONS
%% Policy update by EGM
function [C,Kprime] = EGM(C,mutil,invmutil,par,mpar,P,meshes,gri)
    %% This function iterates forward the consumption policies for the consumption Savings
    % model using the EGM method. C (k x z) is the consumption policy guess. MUTIL and INVMUTIL are
    % the marginal utility functions and its inverse. PAR and MPAR are parameter structures.
    % P is the transition probability matrix. MESHES and GRI are meshes and grids for income
    % (z) and assets (k).

    mu     = mutil(C); % Calculate marginal utility from c'
    emu    = mu*P';     % Calculate expected marginal utility
    Cstar  = invmutil(par.beta *(1+par.r) * emu);     % Calculate cstar(m',z)
    Kstar  = (Cstar  + meshes.k - meshes.z)/(1+par.r); % Calculate mstar(m',z)
    Kprime = meshes.k; % initialze Capital Policy

    for z=1:mpar.nz % For massive problems, this can be done in parallel
        % generate savings function k(z,kstar(k',z))=k'
        Savings     = griddedInterpolant(Kstar(:,z), gri.k,'linear'); %
        Kprime(:,z) = Savings(gri.k);     % Obtain k'(z,k) by interpolation

    end
    BC         = meshes.k<repmat(Kstar(1,:),mpar.nk,1); % Check Borrowing Constraint for all incomes
    % Replace Savings for HH saving at BC
    Kprime(BC) = par.b; % Households with the BC flag choose borrowing contraint
    % generate consumption function c(z,k^*(z,k'))
    C          = meshes.k*(1+par.r)+ meshes.z - Kprime; %Consumption update from constraint
    

end