%% Solve the stochastic growth model using parameterized expectations
%Summary: we are solving 
clear
clc
close all
%% 1. Define parameters


%% Define Economic parameters
par.beta  = 0.95; % Discount factor
par.alpha = 0.5;    % Curvature of production function
par.gamma = 1;    % Coefficient of relative risk aversion
par.delta = 1;    % Depreciation
par.zeta = 1;     %Externality
par.rhoZ = 0.75;  %persistence of technology shock
prob.z     = [0.875, 0.125; 0.125, 0.875];% Transition probabilities for productivity
KSS = (((1-par.beta)/par.beta + par.delta)/par.alpha).^(1/(par.alpha -1 ));
%% Define Numerical Parameters
mpar.nk   = 25;  % number of points on the capital grid
mpar.nK   = 15;   % points on the aggregate capital grid
mpar.nz   = 2;    % number of points on the productivity grid
mpar.mink = KSS*.95;  % lowest point on the capital grid
mpar.maxk = KSS*1.05;    % highest point on the capital grid
mpar.crit = 1e-6; % Precision up to which to solve the value function
mpar.nstates =2;
mpar.ncontrols =2;
%% Produce grids

gri.k = exp(linspace(log(mpar.mink),log(mpar.maxk),mpar.nk)); % individual capital grid
gri.K = exp(linspace(log(mpar.mink),log(mpar.maxk),mpar.nK)); % aggregate capital grid
gri.z = [0.99,1.01];
%% Display Model
TablePar={'Discount Factor:', par.beta; 'Returns to Scale', par.alpha; ...
    'Relative Risk Aversion', par.gamma; 'Depreciation', par.delta};

TablePar
%% 2. Generate grids, Meshes and Income
% Meshes of capital and productivity
[meshes.k, meshes.K, meshes.z]= ndgrid(gri.k,gri.K,gri.z);
Y = meshes.z .*meshes.K.^par.alpha; % Calculate Output  $$Y=zK^{\alpha}$$
R = par.alpha.*meshes.z.*(meshes.K.^(par.alpha-1)); % Calculate Rental Rate $$z\alpha /K^{1-\alpha}$$
Profit = (1 - par.alpha).* Y; %profit is come proportion of the production function
H_map= ones(mpar.nk,mpar.nK,mpar.nz)*(mpar.maxk+mpar.mink)/2; %Starting guess aggregate capital remains constant, half the steady state
%This is a 25x15x2 matrix filled with values equal to half the steady state


%% 3. Define utility functions / marginal utility

if par.gamma ==1
    util     = @(c)log(c); % Utility
    mutil    = @(c) 1./c;  % Marginal utility
    invmutil = @(mu) 1./mu;% inverse marginal utility
else
    util     = @(c) 1/(1-par.gamma).*c.^(1-par.gamma); % Utility
    mutil    = @(c) 1./(c.^par.gamma); % Marginal utility
    invmutil = @(mu) 1./(mu.^(1./par.gamma)); % inverse marginal utility
end

%% Parametrized expectations (interpolant as expectations)
distK=9999;
C     = (Profit + (R-par.delta).*meshes.k); %Initial guess for consumption policy: roll over assets
while distK>mpar.crit
    Cold  = C;
    distEG  = 1; % Initialize Distance
    while distEG>mpar.crit % EGM loop for 3d problem
        C      = EGM_indiv(Cold,mutil,invmutil,R,Profit,par,mpar,prob.z, H_map, meshes,gri); % Update consumption policy by EGM
        %This solves for the consumption sequence of the individual.
        distEG = max(abs(C(:)-Cold(:))); % Calculate Distance
        Cold   = C; % Replace old policy
    end
    [~,kprime] = EGM_indiv(Cold,mutil,invmutil,R,Profit,par,mpar,prob.z,H_map,meshes,gri);
    %Finds the argmaxes to then find the endogenous grid, k_t, which could
    %be off grid of course, made by this interpolant

    % Find individual policy function h(k,K,z)
    h_fun = griddedInterpolant({gri.k ,gri.K, gri.z},kprime); %parameterizing the policy function. The 
%grid could have been the same size, but the idea is to show that its
%applicable to the krusell smith, where the variance for the individual is
%greater, thus a need for more grid points. 
    H_map_new = h_fun(meshes.K,meshes.K,meshes.z); %Let the new aggregate policy equal the ind_policy function
    distK = max(abs(H_map_new(:)-H_map(:)));
    H_map=0.1*H_map_new+0.9*H_map; %Our H_map then becomes this convex combination which we will use for the 
    %next iteration. The idea is place some weight on the new iteration,
    %but most of the weight on the previous iteration.
end
H_map =squeeze(H_map(1,:,:));
%% Direct Attack: Social Planner
% Meshes of capital and productivity
[meshes.K, meshes.z]= ndgrid(gri.K,gri.z);
Y = meshes.z .*meshes.K.^par.alpha; % Calculate Output  $$Y=zK^{\alpha}$$
R = par.alpha.*meshes.z.*(meshes.K.^(par.alpha-1)); % Calculate Rental Rate $$z\alpha /K^{1-\alpha}$$
Profit = (1 - par.alpha).* Y;
distK=9999;
C     = (Profit + (R-par.delta).*meshes.K); %Initial guess for consumption policy: roll over assets
Cold  = C;
distEG  = 1; % Initialize Distance
while distEG>mpar.crit
    C      = EGM_social(Cold,mutil,invmutil,R,Profit,par,mpar,prob.z,meshes,gri); % Update consumption policy by EGM
    distEG = max(abs(C(:)-Cold(:))); % Calculate Distance
    Cold   = C; % Replace old policy
end
[~,H_map_S] = EGM_social(Cold,mutil,invmutil,R,Profit,par,mpar,prob.z,meshes,gri);
%% Graphical output
plot(gri.K,H_map)
hold on
plot(gri.K,H_map_S,'--')

%% 6. Perturbation approach
indexes.Z=1; indexes.K=2; indexes.C=3; indexes.R=4;
% Find steatdy state
% $$R-\delta = (1-\beta)/\beta$$
KSS = (((1-par.beta)/par.beta + par.delta)/par.alpha).^(1/(par.alpha -1 ));
YSS = KSS.^par.alpha;
CSS = YSS - KSS*par.delta;
RSS = 1+ par.alpha*KSS.^(par.alpha -1) - par.delta;
ZSS = 0;
XSS(indexes.Z)=ZSS; XSS(indexes.K)=KSS; XSS(indexes.C)=CSS; XSS(indexes.R)=RSS; 

F = @(XPrime,X) Fsys(XPrime,X,XSS,mpar,par,indexes, mutil);
epsilon=0.0001;
F0 = F(zeros(size(XSS)),zeros(size(XSS))); % Check steady state arror is 0

%Derivatives
A = zeros(mpar.nstates+mpar.ncontrols);
for j=1:(mpar.nstates+mpar.ncontrols)
    h = zeros((mpar.nstates+mpar.ncontrols),1);
    h(j) = epsilon;
    A(:,j) = (F(h , zeros(size(XSS)) ) - F0)/epsilon;
end

B = zeros(mpar.nstates+mpar.ncontrols);
for j=1:(mpar.nstates+mpar.ncontrols)
    h = zeros((mpar.nstates+mpar.ncontrols),1);
    h(j) = epsilon;
    B(:,j) = (F(zeros((mpar.nstates+mpar.ncontrols),1),h ) - F0)/epsilon;
end
[hx, gx]= SGU(A,B,mpar); %smith -g.-uribe
[meshes.K, meshes.z]= ndgrid(gri.K,gri.z);
KPrimeL =reshape(gx(2,:)*[log(meshes.z(:))-ZSS,meshes.K(:)-KSS]',[mpar.nK,mpar.nz]) +KSS;

plot(gri.K,KPrimeL,'-.')
plot(gri.K,gri.K,'k:')
legend({'savings low (PE)', 'savings high (PE)','savings low (SP)','savings high (SP)','savings low (lin)','savings high (in)','45 degree'},'Location','NW')

%% Functions (new stuff)

%% Policy update by EGM (parameterized expectations)
function [C,kprime,KPrime] = EGM_indiv(C,mutil,invmutil,R,Profit,par,mpar,P, KPrime,meshes,gri)
% This function iterates forward the consumption policies for the consumption Savings
% model using the EGM method. C (k x K x z) is the consumption policy guess. MUTIL and INVMUTIL are
% the marginal utility functions and its inverse. PAR and MPAR are parameter structures.
% P is the transition probability matrix. MESHES and GRI are meshes and grids for income
% (z) and assets (k).

mu     = mutil(C); % Calculate marginal utility from c'
% Calculate expected marginal utility times interest as function of K',k',z
emu    = reshape(reshape(par.beta *(1+R-par.delta) .*mu ,[mpar.nK*mpar.nk,mpar.nz])*P',[mpar.nk,mpar.nK,mpar.nz]);
emu_fun=griddedInterpolant({gri.k ,gri.K ,gri.z },emu,'linear'); %still on the grid. 
%Cstar  = invmutil(par.beta.*R.*emu_fun);     % Calculate cstar(K'(K,z), k',z), evaluate for ,y H(K,z)
Cstar  = invmutil(emu_fun(meshes.k, KPrime, meshes.z));     % Calculate cstar(K'(K,z), k',z), evaluate for ,y H(K,z)

kstar  = (Cstar  + meshes.k - Profit)./(1+R-par.delta); % Calculate mstar(m',z)
kprime = meshes.k; % initialze Capital Policy

for z=1:mpar.nz % For massive problems, this can be done in parallel
    for KK=1:mpar.nK
        % generate savings function k(z,kstar(k',z))=k'
        Savings     = griddedInterpolant(kstar(:,KK,z),gri.k, 'linear');
        kprime(:,KK,z) = Savings(gri.k);     % Obtain k'(z,k) by interpolation
    end
end
% Borrowing constraint never binds
% generate consumption function c(z,k^*(z,k'))
C          = meshes.k.*(1+R-par.delta) + Profit - kprime; %Consumption update
end

%zprime to z, then what you have is, you have .. emufun is defined as a tensor of the k-dimension and the z-dimension, 
%savings is not defined, is not coming out of a tensor. Kstar is different on the grid for each z. 
%% Rep Agent model as system of non-linear difference equations, Z is continuous. and above we discretize. rhoZ was chosen to allow 
%them to have the same answer
function F = Fsys(XPrime,X,XSS,mpar,par,indexes, mutil) %At the expectation 
    F = zeros(mpar.nstates+mpar.ncontrols,1);
    % Read out variable values
    C=X(indexes.C)+XSS(indexes.C); CPrime=XPrime(indexes.C)+XSS(indexes.C);
    K=X(indexes.K)+XSS(indexes.K); KPrime=XPrime(indexes.K)+XSS(indexes.K);
    R=X(indexes.R)+XSS(indexes.R); RPrime=XPrime(indexes.R)+XSS(indexes.R);
    Z=X(indexes.Z)+XSS(indexes.Z); ZPrime=XPrime(indexes.Z)+XSS(indexes.Z);
    % Economic model, taking expectations over each one
    F(indexes.C) = mutil(C) - par.beta *RPrime * mutil(CPrime); %Euler Equation 
    F(indexes.R) = R - (1+par.alpha * exp(Z)*K^(par.alpha-1) - par.delta); %interest rate = 1/beta
    F(indexes.Z) =  par.rhoZ.*Z-ZPrime; % Law of Motion for Z
    F(indexes.K) = KPrime - (exp(Z)*K^par.alpha + (1-par.delta)*K) + C; %capital accumulation equation
end


% Derive laws of motion
function [S2C,S2Sprime]=SGU(A,B,mpar)
[s, t, Q, Z] = qz(A,-B); %generalized schur decomppsition, 

relev = abs(diag(s))./abs(diag(t)); %eigenvalues
ll    = sort(relev); %find which ones are larger than 1
slt   = relev>=1;
nk    = sum(slt); %is equal to the number of states? 

if nk>mpar.nstates
    warning(['The Equilibrium is Locally Indeterminate!' ])
elseif nk<mpar.nstates
    warning(['No Local Equilibrium Exists!'])
end
[s,t,~,Z] = ordqz(s,t,Q,Z,slt); %reorder

z21=Z(nk+1:end,1:nk);
z11=Z(1:nk,1:nk);
s11=s(1:nk,1:nk);
t11=t(1:nk,1:nk);

%Checks
if rank(z11)<nk
    warning('invertibility condition violated')
end
z11i=z11\eye(nk);
S2C=real(z21*z11i); %States2Controls, hx
S2Sprime=real(z11*(s11\t11)*z11i); %LOM states, gx
end

%% Functions (old stuff)

function [C,Kprime] = EGM_social(C,mutil,invmutil,R,Profit,par,mpar,P,meshes,gri)
%% This function iterates forward the consumption policies for the consumption Savings
% model using the EGM method. C (K x z) is the consumption policy guess. MUTIL and INVMUTIL are
% the marginal utility functions and its inverse. PAR and MPAR are parameter structures.
% P is the transition probability matrix. MESHES and GRI are meshes and grids for income
% (z) and assets (k).

mu     = mutil(C); % Calculate marginal utility from c'
emu    = ((1+R-par.delta).*mu)*P';     % Calculate expected marginal utility
Cstar  = invmutil(par.beta* emu);     % Calculate cstar(m',z)
Kstar  = (Cstar  + meshes.K - Profit)./(1+R-par.delta); % Calculate mstar(m',z)
Kprime = meshes.K; % initialze Capital Policy

for z=1:mpar.nz % For massive problems, this can be done in parallel
    % generate savings function k(z,kstar(k',z))=k'
    Savings     = griddedInterpolant(Kstar(:,z),gri.K,'linear');
    Kprime(:,z) = Savings(gri.K);     % Obtain k'(z,k) by interpolation
    
end
C          = meshes.K.*(1+R-par.delta)+ Profit - Kprime; %Consumption update
end
