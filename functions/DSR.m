function [dsr] = DSR(R, method)
    
% calculates dsr from return series R
% wrapper on PSR
%
% inputs
% ---------------
% R         - matrix of returns
% SR_b      - benchmark sharpe ratio 
% method    - method to determine number of independent trials

    
% psr constants
T = size(R,1);
g_3 = skewness(R);
g_4 = kurtosis(R);
SR = sharpe(R);





% dsr constants - temp; need to adjust
switch method
    case 'std'
        N = size(R,2);
        var_SR = var(SR);
    case 'ONC'
        
        k=2:200; % number of clusters to try

        myfunc = @(X,K)(kmeans(X, K)); 
        eva = evalclusters(R, myfunc,'silhouette','klist',k);
        classes = kmeans(R, eva.OptimalK);
        N = eva.OptimalK;

        for k=1:N
          pos{k} = find(classes==k);% find position of returns in cluster k
          C{k} = R(pos{k},:); % set of strategies in cluster k
          Sigma{k} = cov(C{k}'); % covariance matrix restricted to strategies in cluster k
          pwgt{k} = 1./diag(Sigma{k}); % A.4. MINIMUM VARIANCE ALLOCATION "one could choose to approximate the weights by setting w proportional to 1/sigma^2 as is typically done in inverse variance allocations"
          pwgt{k} = pwgt{k}/sum(pwgt{k});
          S{k} = pwgt{k}'*C{k};
          Ann_SR{k} = (mean(S{k}))/sqrt(var(S{k}));
        end

        Ann_SR_k = cell2mat(Ann_SR);
        var_SR = var(Ann_SR_k);
        
        
    case 'fSPC'
        N = size(R,2);
        var_SR = var(SR);
end

em = double(eulergamma);
SR_b = sqrt(var_SR)*((1-em)*norminv(1 - 1/N) + em*norminv(1-1/N*exp(-1)));


% psr calc using standard norm cdf
sigma = sqrt(1-g_3.*SR + (g_4-1)./4.*SR.^2)/sqrt(T-1);
mu = SR;

dsr = 1 - normcdf(SR_b, mu, sigma);