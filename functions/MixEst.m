function [pi_m, Var_pi] = MixEst(pi1, pi2, Sigma_pi1, Sigma_pi2)

% Mixed Estimation 3 
% incl. adjustments for NaNs

% Inputs
% pi_1, pi_2        - vector of mean estimates
% Sigma_1, Sigma_2  - uncertainty matrices

% nan mgmt
nan_idx = (or(isnan(pi1),isnan(pi2)));
pi1(nan_idx) = 0;
pi2(nan_idx) = 0;

nan_IDX = logical(double(~nan_idx)*double(~nan_idx'));
Sigma_pi1(~nan_IDX) = 0;
Sigma_pi2(~nan_IDX) = 0;

% other NaNs
Sigma_pi1(isnan(Sigma_pi1)) = 0;
Sigma_pi2(isnan(Sigma_pi2)) = 0;




% size
n = length(pi1);
% means
y = [pi1;pi2];
% block covariance 
Z = zeros(n);
W = [(diag(diag(Sigma_pi2))) Z ; Z (diag(diag(Sigma_pi1)))];
% coefficients
I = eye(n);
A = [I;I];

% calculate estimate
pi_m = pinv(A'*W*A)*(A'*W*y);
% calculate variance of estimator
Var_pi = diag(diag(Sigma_pi2))*pinv(A'*W*A)*diag(diag(Sigma_pi1));

% put nans back
pi_m(pi_m==0) = NaN;
Var_pi(Var_pi==0) = NaN;


