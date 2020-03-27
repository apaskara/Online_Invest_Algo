function [W_g, invSig] = GMV_001ac(Sigma, type)
% Inputs:
% Sigma - Covariance matrix
% type - 'cl' : clean covariance matrix using RMT
%        'pinv' : psuedo-inverse
%        'inv' : matrix inverse  
%
% version: 001ab - gives nan weights output for Sigma = NaN, uses
% naninv_001ac
%
% NEED to include constraints, 
% calculates the global minimum variance portfolio using risk model
% (covariance model). Allows for linear equality constraints.
% 
% TBDL:
% 

% find nans
nanind = isnan(Sigma);
% invert convariance matrix
[invSig] = naninv(Sigma, type);
% set NaN = 0,
invSig(nanind) = 0;
% vector of ones
one = ones(size(invSig,1),1);
% solve for GMV weights
W_g = invSig*one/(one'*invSig*one);
% restore NaNs in invSig
invSig(nanind) = NaN;
% restore NaNs in weights
nandiag = isnan(diag(Sigma));
W_g(nandiag) = NaN;