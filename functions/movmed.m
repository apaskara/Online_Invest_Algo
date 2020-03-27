function [Med] = movmed(A,x,nanflag)
% movmed calculates the median over rolling window of observations.
% median is calculated along rows, ie each column is a different series.
% nanflag indicates if nans should be included or excluded from sample to
% calculate mean
% 
% see also: median
%
[r c] = size(A);
% pre-allocate
Med = NaN*ones(r,c);

for i = 1:r,
    % look back
    lb = min(i-1,x(1));
    % look forward 
    lf = min(r-i,x(2));
    % calc median omitting nan
    Med(i,:) = median(A(i-lb:i+lf,:),nanflag);
end;    
    