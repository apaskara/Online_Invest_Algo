function [s] = online_ts_reg(xdata, ydata, const, varargin)
% Curent Situation:
% This function performs a recursive regression for each output signal in
% ydata onto the input signals xdata. 
% The parameters are caluclated by recursively calculating the normal
% equation: 
% B1 = inv(X1'X1)(X1'y1)
% B1 = {inv(X0'X0) + C(inv(x1'x1))}{(X0'y0 + lambda*weight*x1'y1)}
% Usually B1 is calculated as:
% B1 = B0 + gain*error
% This has the advantage of controlling the gain directly
% 
% The weight function decreases the weights of new observations that deviate
% from the current estimate. ie the weight is a function of the apriori
% error
%
% TBDL: 
%   NB: recursively calculate deviation from median - DONE
%   change calculation method to B1 = B0 + delta if necessary.
%   include bisquare weight function
%   online formulation - ie include arguments for previous estimates
% 
% INPUTS
% --------------
% xdata     (req)   -   2-dimensional array unfiltered input
% ydata     (req)   -   2-d array of output signals
% const     (req)   -   {'y','n'}. include regression constant?
% lambda    (opt)   -   scalar. memory factor
% arg_weght (opt)   -   {'huber','std'} robust weight functions
% arg_invR0 (opt)   -   initial inv(X'X)
% arg_p0    (opt)   -   initial (X'y)
% arg_B0    (opt)   -   initial beta
% 
% REFERENCES:
% 



%default parms for optional aguments
lambda          = 0.98;
arg_weight      = 'huber';
c               = 999999;
if const       == 'y'
    xdata       = [ones(size(xdata,1),1) xdata(:,:)];
end
arg_invR0       = c*eye(size(xdata,2));
arg_p0          = zeros(size(xdata,2), size(ydata,2));
        
%update default parms with user inputs
nvarargin = numel(varargin);
if nvarargin > 0
    for i=1:nvarargin
        switch varargin{i}{1,1}
            case 'lambda'
                lambda  = varargin{i}{1,2};
            case 'p0'
                arg_p0  = varargin{i}{1,2};
            case 'invR0' 
                arg_invR0   = varargin{i}{1,2};
            case 'B0' 
                arg_B0  = varargin{i}{1,2};
                arg_p0  = arg_invR0 \ arg_B0;
            case 'weight'
                arg_weight  = varargin{i}{1,2};
        end
    end
end

%% Estimate median average deviation estimator of Residual standard deviation
% whole sample
% MdAD = mad(ydata,1);
% moving median
wl = round(1/(1-lambda));
MdAD = movmed(ydata,[wl 0],'omitnan');
% recursive update for determining median


%% weight functions 
switch arg_weight
    case 'std' %no robust adjustment
        w = @(e1,MAD)1;
    case 'huber'	
        k = 1.385; % tuning constant
        w = @(e1,MAD)(min(1,k/(abs(e1/MAD))));
end

%% adjust dataset for NaNs - 
%set 0s to NaN
xdata(xdata==0) = NaN;
ydata(ydata==0) = NaN;
%set NaN rows to zero in body of code => beta stays constant.
% count number of NaNs for each company
sumnan = nansum(~isnan(ydata),1);

%% recursive regression
% run regression for each company
for i = 1:size(ydata,2) 
            y = ydata(:,i);
            x = xdata;
            % identify rows with 
            nidx = (isnan(y) | transpose(sum(transpose(isnan(x)))>0));
            % set NaN to zero
            y(nidx,:) = 0;
            x(nidx,:) = 0;
            % initial invR0
            invR0 = arg_invR0;
            % initial p0
            p0 = arg_p0(:,i);
            % initial B0
            B0 = invR0 * p0;
            for j = 1:size(ydata,1) %every month          
                % new response      
                y1 = y(j); 
                % new input
                x1 = x(j,:);
                % calculate a priori error
                e1 = y1 - x1*B0;
                % calculate robust weight
                weight(j,i) = w(e1,MdAD(j,i));
                % update inverse of autocorrelation matrix
                invR1 = (1/lambda)*(invR0 - invR0*x1'*inv(x1*invR0*x1'+lambda*inv(weight(j,i)))*x1*invR0);
                % update autocorrelation vector
                p1 = lambda*p0 + weight(j,i)*x1'*y1;
                % update beta
                B1 = invR1*p1;
                % stats
                s.beta(j,i,:)       = B1;
                s.y_hat(j,i)        = x1*B1;
                s.res(j,i)          = y(j) - s.y_hat(j,i);
                s.weight(j,i)       = weight(j,i);
%                 s.B0(j,i,:)         = B0;
                s.fc_err(j,i)       = e1;
%                 s.p0{j,i}           = p0;
%                 s.p0{j,i}           = p1;
%                 s.invR0{j,i}        = invR0;
%                 s.invR1{j,i}        = invR1;
                
                % testing stat
%                 s.huber{j,i}        = k/(abs(e1/MdAD(j,i)));
                % update for next iteration 
                invR0 = invR1;
                p0 = p1;
                B0 = B1;
            end
            s.res(nidx,i) = NaN;
            s.fc_err(nidx,i) = NaN;
end
sumnan = sum(~isnan(s.res));
sqRES = nanmean(s.res.*s.res,1);
s.MSE = nansum(sqRES.*sumnan)/sum(sumnan);
s.MFE = nansum(nanmean(s.fc_err.*s.fc_err,1).*sumnan)/sum(sumnan);
s.invR0 = invR0;

%% temp fix
s.beta(s.beta == 0) = NaN;
% s.varargin          = varargin;
