function [MU,mu_T] = ewma_mean(lambda, xdata, varargin)
% exponentially weighted recursive estimation of mean
% applies zero-order hold on mean for nans
%
% INPUTS
% -------
% lambda is forgetting factor
% xdata is matrix of sequential data with earliest data at top
% mu0 (vararg) is initial mean
%
% OUTPUTS
% --------
% mu_T is vector of last mean
% MU is matrix of EWMA
%



% find size of data matrix
[T,a] = size(xdata);
% make sure that initial means have same number of columns to xdata
if numel(varargin) > 0,
    % initial means
    mu0 = varargin{1};
else
    % set to NaN if mean not specified
    mu0 = NaN*ones(1,a);
end;

%% EWMA calc
% pre-allocate MU
MU = NaN*ones(T+1,a);
% initialise ewma mean
MU(1,:) = mu0;
% at each period
for i = 1:T,
   
    % initialize at each point
%     curr_nan = zeros(1,size(MU,2));
    % find nans in current 
    curr_nan = isnan(MU(i,:));
    % clean index
%     inc_nan = zeros(1,size(xdata,2))
    % find nans
    inc_nan = isnan(xdata(i,:));
    
    % case 1: nan prior mean, nan incoming data => NaN post mean
    case1_nan = curr_nan & inc_nan;
    MU(i+1,case1_nan) =  NaN;
    % case 2: nan prior mean, incoming data => post mean = inc data
    case2_nan = (curr_nan==1) & (inc_nan==0);
    MU(i+1,case2_nan) = xdata(i,case2_nan);
    % case 3: prior mean but nan in incoming data
    case3_nan = (curr_nan==0) & (inc_nan==1);
    MU(i+1,case3_nan) = MU(i,case3_nan);    
    % case 4: prior mean and incoming data
    case4_nan = (curr_nan==0) & (inc_nan==0);
    MU(i+1,case4_nan) = (1-lambda)*xdata(i,case4_nan) + lambda*MU(i,case4_nan);
    
    
end;
% drop initial means
MU = MU(2:end,:);
% most recent mean
mu_T = MU(T,:);

