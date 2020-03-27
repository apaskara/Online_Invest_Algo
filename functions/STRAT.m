function [W_s, invCov, condno] = STRAT_001aaa(Sigma, mu, g, inv, varargin)
% NEED TO INCLUDE CONSTRAINTS
% function calculates strategic benchmark weights. Allows for linear
% equality constraints and mean estimate uncertainty (var(mu)).
% Note:
% if active risk-return then W_s'1 = 0
% if total risk-return then (W_s + W_t)'1 = 0
% 
% What to do if we have a mean without var/cov estimates? 
% for now assume that these are both populated or not.
% also assume that the corresponding uncertainty is NaN.
% if we have mean, but no uncertainty, assume mean is certain
%   
%
% TBDL:
%   1. include constraints
%   2. remove gamma as parameter to incorporate constraints
%   3. without leverage constraints, weights are enormous

% check that means and variance matrix have compatible size
if length(mu) ~= size(Sigma,1)
    error('size of means vector does not match covariance matrix');
end;

% Var(mu) - variance of mean estimator
if numel(varargin) > 0, 
    Sig_mu = varargin{1};
    if size(Sig_mu)~=size(Sigma),
        error('size of uncertainty matrix for means does not match covariance matrix')
    end;        
else 
    Sig_mu = zeros(size(Sigma));
end;

% find nans in means
nanind_mu = isnan(mu);
% find nans in variances - diagonals
nanind_Sig = isnan(diag(Sigma));
% nan in mean or variance
nanind = nanind_mu | nanind_Sig;
% fill with nans
mu(nanind) = NaN;
% Sigma(nanind, nanind') = NaN;
% Sig_mu(nanind, nanind') = NaN;

NANind = (~(~nanind*~nanind'));
Sigma(NANind) = NaN;
Sig_mu(NANind) = NaN;

% marginal covariance matrix (marginal density over parameter space)
Cov = Sigma + Sig_mu;
% invert covariance matrix
[invCov] = naninv(Cov,inv);
condno = NaN;
% set NaNs to zero
invCov(isnan(invCov)) = 0;
mu(nanind) = 0;
% vector of ones
one = ones(size(mu));

% calculate scaled portfolio weights
% tmp = (mu*one' - one*mu');
% tmp1 = (mu*one' - one*mu')*one;
% tmp2 = (invCov*(mu*one' - one*mu')*invCov)*one;

w_s = 1/(one'*invCov*one)*(invCov*(mu*one' - one*mu')*invCov)*one;
% TEMP: calcualte gamma based on leverage constraint (temporary)
% lev = 1; %leverage
% g = (abs(w_s)'*one)/lev; 
% g = 1;

% g = one'*invCov*mu;
% calculate tactical weights
W_s = (1/g)*w_s;
% restore nans
W_s(nanind) = NaN;

