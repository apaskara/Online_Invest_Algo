function [psr] = PSR(R, SR_b)
    
% calculates psr from return series R
    
    % psr constants
    T = length(R);
    g_3 = skewness(R);
    g_4 = kurtosis(R);
    SR = sharpe(R);

    % psr calc using standard norm cdf
    sigma = sqrt(1-g_3.*SR + (g_4-1)./4.*SR.^2)/sqrt(T-1);
    mu = SR;

psr = 1 - normcdf(SR_b, mu, sigma);

    
