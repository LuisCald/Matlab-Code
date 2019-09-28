%% Huggett's Model: Calculating the equilibrium interest rate 
%% Aiyagari Model
clear
clc
close all
%% 1. Define parameters

mpar.nk   = 100;   % Number of points on the asset/bond grid
mpar.nz   = 2;    % Number of points on the log-productivity grid
mpar.crit = 1e-5; % Numerical precision
mpar.maxk = 6;    % Maximimum assets
mpar.mink = -9/4;    % Minimum Assets (equal to Borrowing Limit)
mpar.T    = 100000;
disp('Numerical parameters')
mpar % Display numerical parameters
% Economic Parameters
par.r     = 0;% Real Rate
par.gamma = 1;    % Coeffcient of relative risk aversion
par.beta  = 0.95; % Discount factor
par.alpha  = 0.36; % Capital Share
par.delta  =0.1;
par.b     = mpar.mink; % Borrowing Limit
disp('Economic parameters')
par % Display economic parameters

%% 2. Generate grids, Meshes and Income
gri.k   = exp(linspace(log(1),log(mpar.maxk-(mpar.mink-1)),mpar.nk))+mpar.mink-1; %Define asset grid on log-linearspaced
prob.z  = [3/5, 2/5; 4/90,  86/90];
gri.z   = [1/9, 10/9];
% Meshes of capital and productivity
[meshes.k,  meshes.z] = ndgrid(gri.k,gri.z);
Y =meshes.z+(1-par.delta)*meshes.k; % Cash at hand (Labor income plus assets cum dividend)

%% 3. Define utility functions

if par.gamma ==1
    util     = @(c)log(c); % Utility
    mutil    = @(c)1./(c);  % Marginal utility
    invmutil =@(mu)1./(mu) ;% inverse marginal utility
else
    util     = @(c) 1/(1-par.gamma).*c.^(1-par.gamma); % Utility
    mutil    = @(c) 1./(c.^par.gamma); % Marginal utility
    invmutil = @(mu) 1./(mu.^(1./par.gamma)); % inverse marginal utility
end
%% Solve for the equilibrium (Ex12), no capital in this economy 
ExcessDemand  = @(R) (K_Agg(R ,1,0,mutil,invmutil,par,mpar,prob.z,meshes,gri));
Rstar_Huggett = fzero(ExcessDemand,[0.02,((1-par.beta)/par.beta)-.005 ])
%% Solve for the equilibrium (Ex12&13)
% $$R+\delta =\alpha K ^{\alpha-1}  N^{1-\alpha} = MPK$$
% $$K^d = \left(\frac{R+ \delta }{\alpha}\right)^{\frac{1}{\alpha-1}} N5 $$
% $$w =(1-\alpha) K ^{\alpha}  N^{-\alpha} = MPL$$

par.gamma=4;
if par.gamma ==1
    util     = @(c)log(c); % Utility
    mutil    = @(c) 1./c;  % Marginal utility
    invmutil = @(mu) 1./mu;% inverse marginal utility
else
    util     = @(c) 1/(1-par.gamma).*c.^(1-par.gamma); % Utility
    mutil    = @(c) 1./(c.^par.gamma); % Marginal utility
    invmutil = @(mu) 1./(mu.^(1./par.gamma)); % inverse marginal utility
end

aux =prob.z^1000; % the long term distribution or the invariant distribution 
N = sum(aux(1,:).*gri.z); %average labor supply. Takes the first row of the invariant matrix "aux", multiplies it by the grid of income states
%to generate the expected value of labor for this state or rather, the
%average labor supply. 
Kdemand = @(R) (N * (par.alpha/(R+par.delta)).^(1/(1-par.alpha))); 
%Solving for K in the R = MPK-delta
rate = @(K) (par.alpha* N.^(1-par.alpha) * K.^(par.alpha-1) -par.delta); 
%MPK minus depreciation = net return on capital, functions of capital
wage= @(K) ((1-par.alpha)* N.^(-par.alpha) * K.^(par.alpha)); 
%simply the MPL, functions of capital
ExcessDemand  = @(K) (K_Agg(rate(K),wage(K),0,mutil,invmutil,par,mpar,prob.z,meshes,gri) - K);
Rstar_Aiyagari = rate(fzero(ExcessDemand, [Kdemand(.01), Kdemand(.045)]))

%% Solve for the equilibrium (Ex14)
% $$R+\delta =\alpha K ^{\alpha-1}  N^{1-\alpha}$$
% $$\left(\frac{R+ \delta }{\alpha}\right)^{\frac{1}{\alpha-1}} N =K$$
% $$w =(1-\alpha) K ^{\alpha}  N^{-\alpha}$$
% Tax rate 0.2
%lok into government debt playing a role
tax = 0.2;
transfer=@(K) tax*wage(K)*N; %rebated lump sum to the households. The transfer is the tax rate times your income, which is 
%expressed as the wage times the efficiency units. 
ExcessDemand  = @(K) (K_Agg(rate(K) ,wage(K) ,transfer(K) ,mutil,invmutil,par,mpar,prob.z,meshes,gri) - K);
Rstar_Aiyagari = rate(fzero(ExcessDemand, [Kdemand(.01), Kdemand(.045)]))

%% SUBFUNCTIONS
%% Asset holdings given r

function [K,kprime,marginal_k]= K_Agg(interest,wage,transfer,mutil,invmutil,par,mpar,P,meshes,gri)
par.r = interest
gri.z = gri.z*wage+transfer;
% labor and transfer income (grid)
[meshes.k, meshes.z] = ndgrid(gri.k, gri.z) ; % labor and transfer income (mesh)
C     = (par.r.*meshes.k+meshes.z); %Initial guess for consumption policy: roll over assets, must b feasible, strictly increasing
Cold  = C; % Save old policy
distEG  = 1; % Initialize Distance
iterEG  = 1; % Initialize Iteration count
while distEG>mpar.crit
    C      = EGM(Cold,mutil,invmutil,par,mpar,P,meshes,gri); % Update consumption policy by EGM
    dd     = max(abs(C(:)-Cold(:))); % Calculate Distance
    
    Cold   = C; % Replace old policy
    iterEG = iterEG+1; %count iterations
    distEG(iterEG) = dd;
end
[~,kprime] = EGM(C,mutil,invmutil,par,mpar,P,meshes,gri);

%% 7. Obtain asset holdings by defining a Markov chain with policies and grid
[~,idk]                 = histc(kprime,gri.k); % find the next lowest point on grid for policy
idk(kprime<=gri.k(1))   = 1; % remain in the index set
idk(kprime>=gri.k(end)) = mpar.nk-1; % remain in the index set
%These three components together form the weights for the 
distance    = kprime-gri.k(idk);
weightright = distance./(gri.k(idk+1)-gri.k(idk));
weightleft  = 1-weightright;
%To remember this part, look at the loops and always to remember to
%consider adding the indices
Trans_array = zeros(mpar.nk,mpar.nz,mpar.nk,mpar.nz); %Assets now, Income now, Assets next, Income next
for zz=1:mpar.nz % all current income states
    for kk=1:mpar.nk % all current asset states
        Trans_array(kk,zz,idk(kk,zz),:)   = weightleft(kk,zz) *reshape(P(zz,:),[1 1 1 mpar.nz]);
        Trans_array(kk,zz,idk(kk,zz)+1,:) =  weightright(kk,zz)*reshape(P(zz,:),[1 1 1 mpar.nz]);
    end
end
Gamma=reshape(Trans_array,[mpar.nk*mpar.nz, mpar.nk*mpar.nz]);
[x,~]=eigs(Gamma',1); % x_{t+1} = x_t P_{K} %eigenvector
x=x./sum(x); %must sum to 1
marginal_k=sum(reshape(x,[mpar.nk, mpar.nz]),2); % wealth distribution, which was solved by the left unit eigenvector
K= sum(marginal_k'.*gri.k) %supply of funds from the steady state distribution. 
end

%% Policy update by EGM
function [C,Kprime] = EGM(C,mutil,invmutil,par,mpar,P,meshes,gri)
%% This function iterates forward the consumption policies for the consumption Savings
% model using the EGM method. C (k x z) is the consumption policy guess. MUTIL and INVMUTIL are
% the marginal utility functions and its inverse. PAR and MPAR are parameter structures.
% P is the transition probability matrix. MESHES and GRI are meshes and grids for income
% (z) and assets (k).

mu     = mutil(C); % Calculate marginal utility from c'
emu    = mu*P';     % Calculate expected marginal utility
  Cstar  = invmutil(par.beta *(1+par.r) * emu);     % Calculate cstar(m',z) from inverse 
    Kstar  = (Cstar  + meshes.k - meshes.z)/(1+par.r); % Calculate kstar(m',z) from resource constraint, capital today
    Kprime = meshes.k; % initialze Capital Policy

for z=1:mpar.nz % For massive problems, this can be done in parallel
    % generate savings function k(z,kstar(k',z))=k'
    Savings     = griddedInterpolant(Kstar(:,z),gri.k ,'linear'); %remember that the "z"'s are the columns
    Kprime(:,z) = Savings(gri.k);     % Obtain k'(z,k) by interpolation
    
end
BC         = meshes.k<repmat(Kstar(1,:),mpar.nk,1); % Check Borrowing Constraint
% Replace Savings for HH saving at BC
Kprime(BC) = par.b; % Households with the BC flag choose borrowing contraint
% generate consumption function c(z,k^*(z,k'))
C          =   meshes.k*(1+par.r)+ meshes.z - Kprime; %Consumption update

% if r is greater than the time preference rate, then households save, then if ther ewas no mlimit to their savings then their policies will always be increasing/ So this means, 
% even at the largest capita;stock, your capital choice tomorrow is outside your grid, meaning the weigh right will be negative extrapolation, so negative entires
% thus the eigenvector is not a list of real numbers. 
%So the interest rate must be smaller than the time preference rate. 
%
end