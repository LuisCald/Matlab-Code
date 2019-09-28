%% Multi-grid
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

grid.k = exp(linspace(log(mpar.mink),log(mpar.maxk),mpar.nk));
grid.z = [0.9,1.1];
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


points = [10 20 30];

for ss=1:length(points)
    tic
    mpar.nk=points(ss);
    %mpar.nz=points(ss);
    %% Generate grids
    grid.k = exp(linspace(log(mpar.mink),log(mpar.maxk),mpar.nk)); %Define asset grid on log-linearspaced
    
    %% Meshes and Cash at Hand (Y)
    [meshes.k,  meshes.z]= ndgrid(grid.k,grid.z);
    Y = meshes.z.*meshes.k.^par.alpha + (1-par.delta).*meshes.k;
    
    %% Initialize Value and Policy Functions
    V      = zeros(mpar.nk,mpar.nz);
    kprime = repmat(grid.k(:),[1,mpar.nz]);
    Vnew   = zeros(mpar.nk,mpar.nz);
    
    %% Value Function Iteration
    dist   = 9999;
    count  = 1;
    tic
    while dist(count)>mpar.crit
        count       = count+1;                % count the number of iterations
        [Vnew,kprime]        = VFI_update_lin(V,Y,util,par,mpar,grid,prob);
        dist(count) = max(abs(Vnew(:)-V(:))); % Calculate distance between old guess and update
        V           = squeeze(Vnew);          % Copy update to value function
    end
    its_plainVFI(ss) = count;
    time1(ss)=toc; %This is the amount of time it takes for the value function to run under different grids
    VFI=V;
end

%% Multigrid VFI
mpar.nk=points(1);
V   = zeros(mpar.nk,mpar.nz);
grid.k = exp(linspace(log(mpar.mink),log(mpar.maxk),mpar.nk)); %Define asset grid on log-linearspaced
for ss=1:length(points)
    tic
    mpar.nk=points(ss);
    %mpar.nz=points(ss);
    %% Generate grids
    gridold.k = grid.k;
    grid.k    = exp(linspace(log(mpar.mink),log(mpar.maxk),mpar.nk)); %Define asset grid on log-linearspaced as we iterate
    F         = griddedInterpolant({gridold.k,grid.z},V); %On the space gridold and grid z, we have evaluations V. 
        %gridold is used in the first step and we derive the Value
        %function through linear interpolation to input in the next
        %iteration.
    V         = F({grid.k,grid.z});

    %% Meshes and Cash at Hand (Y)
    [meshes.k,  meshes.z]= ndgrid(grid.k,grid.z); %two 100x2's
    Y = meshes.z.*meshes.k.^par.alpha + (1-par.delta).*meshes.k;
    
    %% Initialize Value and Policy Functions
    kprime = repmat(grid.k(:),[1,mpar.nz]);  %This is a 100x1 multiplied by 1x2 = 100x2
    
    %% Value Function Iteration
%Now  using the V above, we have a better guess. Our guess gets better with
%more grid points and thus, less of a need for iterations. 
    dist   = 9999;
    count  = 1;
    tic
    while dist(count)>mpar.crit
        count         = count+1;                % count the number of iterations
        [Vnew,kprime] = VFI_update_lin(V,Y,util,par,mpar,grid,prob);
        dist(count)   = max(abs(Vnew(:)-V(:))); % Calculate distance between old guess and update
        V             = reshape(Vnew,[mpar.nk, mpar.nz]);          % Copy update to value function
    end
    its_multiVFI(ss) = count;
    time2(ss)=toc;
end
%%
disp('running times (pVFI/mVFI)')
disp(time1)
disp(cumsum(time2)) %It takes 16 seconds total! to run through 3 grids. Rather, with VFI, 28 seconds!
                       %In addition, as we increase grids, the time it
                       %takes to process does not change, meaning increased
                       %efficiency for Multi-grid.
disp('iterations (pVFI/mVFI)')
disp(its_plainVFI) %for plain VFI, we start with same guess, increasing grid size
disp(its_multiVFI) %For multi-grid, where we start with better guesses. 

%% sub-functions
function [Vnew,kprime] = VFI_update_lin(V,Y,util,par,mpar,grid,prob)
V=reshape(V,[mpar.nk,mpar.nz]);
kprime = zeros(size(V));
Vnew   = zeros(size(V));
EV     = par.beta* V* prob.z';   % Calculate expected continuation value

for zz=1:mpar.nz
    ev_int= griddedInterpolant({grid.k},EV(:,zz),'linear');
    for kk=1:mpar.nk
        f             = @(k)(-util(Y(kk,zz)-k)-ev_int(k));
        [kp,v]        = fminbnd(f,0,Y(kk,zz));
        Vnew(kk,zz)   = -v;
        kprime(kk,zz) = kp;
    end
end
Vnew=Vnew(:);
end

