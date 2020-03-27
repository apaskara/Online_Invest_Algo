function [COV_T, ewCOV, MU] = ewma_cov(lambda, xdata, varargin)
%
% version:  001ae   - include parameter to seperate data into
% initialization period and test period
               
%
% ewmacov calculates the exponentially weighted covariance matrix for panel
% data. ewmacov applies zero-order-hold on covariances if data is nan (for now). 
% lambda = 1 should recover the rolling window covariance matrix.
% 
% ewma_cov is a large sample approximation to the offline case. It will not
% produce the same estimate as a batch exponentially-weighted estimate. 
%
% what mean should be used? here we use the exponentially weighted moving
% average.
% 
% NB: if the length of each time series is different in the data matrix,
% then it is possible that the resulting covariance matrix will not be
% positive definite. the asset with data will have a variance estimate,
% whilst the other asset(s) will not. when the first pairwise deviation is
% calculated, if the prior estimate of the asset's variance is less than
% current squared deviation the posterior variance will be smaller than the
% prior. However, the other elements of the posterior covariance will be
% equal to the current deviation matrix. The resulting matrix is not
% positive definite. (for illustration take 2x2 case)
%
% prior Cov             current Dev                 post Cov
%   N | N               0.09 | (0.3)(0.2)           0.09 | (0.3)(0.2) 
%   ------              ----------------            ----------------  
%   N | 0.03       (0.3)(0.2)| 0.04            (0.3)(0.2)| a0.03 + (1-a)0.04
%
% The deviation matrix is singular (rank 1), whilst the post Cov has
% negative determinant
%
% INPUTS
% ------
% lambda        - forgetting factor [0,1].
% xdata         - panel data, multiple time series, top row is 1st obs,
%                   bottom row is latest
% COV_0 (opt)   - initial covariance matrix
% mu0 (opt)     - initial mean 
%
% variable arguments are entered as cell arrays with 2 elements where the
% first element is the name of the parameter eg {'mu0',xxx}
%
% OUTPUTS
% -------
% COV_T         - latest covariance matrix (time T)
% COV           - 3d matrix. covariance matrices over time
% MU            - matrix of exponentially weighted means
%
% TBDL
%   1. ideally, want robust estimate of mean and covariance
%   
%% default initialization 
mu_0 = NaN*ones(1,size(xdata,2));
COV_0 = NaN*ones(size(xdata,2),size(xdata,2));
p = 0;

%% load optional variables (initialize variables)
if numel(varargin) > 0
    for i = 1:numel(varargin),
        switch varargin{i}{1}
            case 'mu0'
                mu_0 = varargin{i}{2};
            case 'COV_0'
                COV_0 = varargin{i}{2};
            case 'trdata'
                p = round(varargin{i}{2}*size(xdata,1));
        end; % switch
    end; % for
end; % if

% pre-allocate memory
[T,a] = size(xdata);
ewCOV = cell(T+1,1);
pw_DEV = cell(T+1,1);
dev = NaN*ones(T+1,a);
MU = NaN*ones(T+1,a);

%% calculate covariances
if p == 0

    % calculate means
    [MU,~] = ewma_mean(lambda, xdata, mu_0);
    % initialize ewma covariance
    ewCOV{1} = COV_0;
    % initialize ewma means
    MU = [mu_0; MU];
    xdata = [NaN*ones(1,a); xdata];
    % calculate ew cov matrix
    for t = 1:T           
            % find nans in current covariance
            curr_nan = isnan(ewCOV{t});
            % find nans in incoming deviations 
            % calculate deviations
            dev(t+1,:) = xdata(t+1,:)-MU(t+1,:);
            % calculate pairwise deviations
            pw_DEV{t+1}(:,:) = dev(t+1,:)'*dev(t+1,:);
            % find nans in incoming data
            inc_nan = isnan(pw_DEV{t+1}(:,:));

            % case 1: nan prior cov, nan incoming data => nan post Cov
            case1_nan = curr_nan & inc_nan;
            ewCOV{t+1}(case1_nan) = NaN;
            % case 2: nan prior cov, incoming data => post cov = NaN
            case2_nan = (curr_nan==1) & (inc_nan==0);
                % case 2.1: case2_nan = 1, pw_DEV = 0 (first observation has 0 mean)
                case21_nan = (case2_nan==1) & (pw_DEV{t+1}) == 0;    
                ewCOV{t+1}(case21_nan) = NaN;   % need 2 observations to calculate dev
                % case 2.2: case2_nan = 1, pw_DEV ~= 0 (2nd observation)
                case21_nan = (case2_nan==1) & (pw_DEV{t+1}) ~= 0;    
                ewCOV{t+1}(case21_nan) = pw_DEV{t+1}(case21_nan);   % need 2 observations to calculate dev      
            % case 3: prior cov, nan incoming data => post cov = prior cov
            case3_nan = (curr_nan==0) & (inc_nan==1);
            ewCOV{t+1}(case3_nan) = ewCOV{t}(case3_nan);
            % case 4: prior mean and incoming data
            case4_nan = (curr_nan==0) & (inc_nan==0);
            ewCOV{t+1}(case4_nan) = (1-lambda)*pw_DEV{t+1}(case4_nan) + lambda*ewCOV{t}(case4_nan);
            % reshape ewCOV{t+1} from vector to matrix
            ewCOV{t+1} = reshape(ewCOV{t+1},size(xdata,2),size(xdata,2));    
    end;

    %% last covariance matrix
    % COV_T = squeeze(ewCOV(T,:,:));
    COV_T = ewCOV{T};

% if want to initialize using data
elseif p > 0 
    % initialize mean and covariance using p datapoints
    trCOV = nancov(xdata(1:p,:));
    [~,trMu] = ewma_mean(lambda, xdata(1:p,:));
    % calculate
%     [COV_T, ewCOV, MU] = ewma_cov(lambda, xdata(p+1:end,:), {'mu0',trMu}, {'COV_0',trCOV});
%   p+2 since initial values are first element
    [COV_T, ewCOV(p+2:end), MU(p+2:end,:)] = ewma_cov(lambda, xdata(p+1:end,:), {'mu0',trMu}, {'COV_0',trCOV});
end

% remove initial points
ewCOV(1) = [];
pw_DEV(1) = [];
MU = MU(2:end,:);
