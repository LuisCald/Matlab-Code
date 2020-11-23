%% Policy Function Iteration

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
ss=1
for ss=1:length(points)
    mpar.nk=points(ss);
    %mpar.nz=points(ss);
    %% Generate grids
    grid.k = exp(linspace(log(mpar.mink),log(mpar.maxk),mpar.nk)); %Define asset grid on log-linearspaced
    tic
    %% Meshes and Cash at Hand (Y)
    [meshes.k,  meshes.z]= ndgrid(grid.k,grid.z); %Returns a 10x2 for the capital grid and productivity grid.
    Y = meshes.z.*meshes.k.^par.alpha + (1-par.delta).*meshes.k; % available resources
    
    %% (A) Based on Spline Interpolation
    
    %% Initialize Value and Policy Functions
    V      = zeros(mpar.nk,mpar.nz); %Always initialize with zeros of size k-grid and z-grid
    kprime = repmat(grid.k(:),[1,mpar.nz]); %grid.k(:) is a 10x1 and we multiply it by 1x2 = 10x2, 1 for each productivity state
    Vnew   = zeros(mpar.nk,mpar.nz); %100x2
    
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
    its_VFI(ss) = count;
    time1(ss)=toc;
    VFI=V;
   
    %% Initialize Value and Policy Functions
    V      = zeros(mpar.nk,mpar.nz); %Now we start over, so we get a fresh Value function matrix 100x2
    tic
    %% Policy Function Iteration
    dist   = 9999;
    count  = 1;
    while dist(count)>mpar.crit
        count=count+1;
        [~,kprime]        = VFI_update_lin(V,Y,util,par,mpar,grid,prob); %Returns the arg maxes, 10x2, h(n=the iteration number)
        
        [~,idk]                 = histc(kprime,grid.k); % an array the same size as kprime indicating the bin number that 
        % each entry in kprime sorts into. For example, in the first
        % iteration, the kprime are near zero. The first bin of grid
        % k is 0.100-0.1167 in this case. So, because kprime are below 0.100, they
        % fall into this bin (<=0.100), which does not exist. Thus, we have
        % to create the following line to ensure that they indeed fall into
        % at least the smallest grid.
                                                        
        idk(kprime<=grid.k(1))   = 1; %index of kprime. Histc returns bullshit if not corrected for those kprime outside of the grid
        idk(kprime>=grid.k(end)) = mpar.nk-1;  
        distance    = kprime - grid.k(idk); 
% compare (k'-k(id_k)). k(id_k) is the capital choice for the bin number that kprime fell into.
% For example, k'=.1100, which means it falls in the first bin. 
% So 0.1100 - 0.1000 = 0.0100 is the distance from the grid edge. 
        weightright = distance./(grid.k(idk+1)-grid.k(idk)); %This is the weight placed on the transition matrix right. 
        % You can think of this as the slope. where distance = (y'-y) and
        % (grid.k(idk+1)-grid.k(idk)) is the change in x-values, our grid.

        weightleft  = 1-weightright;%This is the weight placed on the transition matrix left
        %The transition matrix shows probabilities from one state to the
        %next. In this case, the capital states are the different states. 
        %Trans = sparse(mpar.nk*mpar.nz,mpar.nk*mpar.nz);

        Trans_array = zeros(mpar.nk,mpar.nz,mpar.nk,mpar.nz); %Assets now, Income now, Assets next, Income next ((10x2)x10)x2
        for zz=1:mpar.nz % all current income states
            for kk=1:mpar.nk % all current asset states
                Trans_array(kk,zz,idk(kk,zz),:)   =  weightleft(kk,zz) *reshape(prob.z(zz,:),[1 1 1 mpar.nz]);
                Trans_array(kk,zz,idk(kk,zz)+1,:) =  weightright(kk,zz)*reshape(prob.z(zz,:),[1 1 1 mpar.nz]);
            end
        end
        Trans=(reshape(Trans_array,[mpar.nk*mpar.nz, mpar.nk*mpar.nz])); % creates the final 20x20
        
        ustar = util(Y-kprime); % utility under current policy. Uses the kprime from above, which used our initial guess.
        % The kprime is the next period optimal stock, which we got from the
        %interpolation
        
        Vnew  = (eye(size(Trans))-par.beta*Trans)\ustar(:); %We take the K' over time and use it as our
        %new guess. This is the sum expressed in matrix form.
        dd    = max(abs(Vnew(:)-V(:)));
        dist(count) = dd; % Calculate distance between old guess and update
        V     = reshape(Vnew,[mpar.nk,mpar.nz]);
    end
    its_PFI(ss)=count;
    time2(ss)=toc;
    differencePFIVFI(ss)=max(abs(VFI(:)-V(:)));
end
%%
disp('running times (VFI/PFI)')
disp(time1)
disp(time2)
disp('iterations (VFI/PFI)')
disp(its_VFI)
disp(its_PFI)
disp('MAD  (VFI/PFI)')
disp(differencePFIVFI)
%% sub-functions
function [Vnew,kprime] = VFI_update_lin(V,Y,util,par,mpar,grid,prob)
V=reshape(V,[mpar.nk,mpar.nz]);
kprime = zeros(size(V));
Vnew   = zeros(size(V));
EV     = par.beta* V* prob.z';   % Calculate expected continuation value

for zz=1:mpar.nz
    ev_int= griddedInterpolant({grid.k},EV(:,zz),'linear');
    for kk=1:mpar.nk
        f             = @(k)(-util(Y(kk,zz)-k)-ev_int(k)); %Multiply everything by -1.
        [kp,v]        = fminbnd(f,0,Y(kk,zz)); %produces the input kp and output v. We take the minimum
        %which is the same as the maximum 
        Vnew(kk,zz)   = -v; %-1*v = positive "v"
        kprime(kk,zz) = kp;
    end
end
Vnew=Vnew(:);
end

