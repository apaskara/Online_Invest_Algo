function [multi] = cs_reg(xdata, ydata, reg_type, lag)
%
% version: 001ab - include lag for x data
%
%
%
% Cross-Sectional Regressions - Estimate characteristic payoffs at each time period
% 
% Author: T Gebbie, A B Paskaramoorthy
%
% Current Situation: Test code needs to be written. But code produces same
% payoffs for the [bias bvtp mv] model from the EPC_HML_SMB_S203_FMP_250 script.
%
% Future Situation: have to write the test code
%
% Function Details:
% -----------------
% {multi} = cs_reg(x_data, y_data, reg_type)
% 
% OUTPUTS
% -------
% s         - struct array containing model inputs, output, paramter
%             estimates and statistics
%
% INPUTS
% ------
% x_data    - 3 dimensional array of company characteristics 
%             (time x company x characteristic)
% y_data    - 2 dimensional array of (excess) company returns
% reg-type  - robustfit, ridge

% default reg_type

%pre-allocate multi
multi.beta       = NaN*ones(size(xdata,1), size(xdata,3)+1); % +1 for constant
multi.se         = NaN*ones(size(xdata,1), size(xdata,3)+1);
multi.t          = NaN*ones(size(xdata,1), size(xdata,3)+1);
multi.adjr2      = NaN*ones(size(xdata,1), 1);
multi.pval       = NaN*ones(size(xdata,1), size(xdata,3)+1);


for t = lag+1:size(ydata,1), % for each time period t
        clear x y;
        x = [squeeze(xdata(t - lag,:,:))]; % x = char data at time i
        x(abs(x)==Inf)=NaN;
        x1 = x;
        y = ydata(t,:)';
        y(abs(y)==Inf)=NaN;
        % set beta and stats to NaN
        beta = NaN*ones(size(x,2),1);
        % find NaNs
        nidx = ~(isnan(y) | transpose(sum(transpose(isnan(x)))>0));
        % remove data without all characteristics (selection effect)
        y=y(nidx);
        x=x(nidx,:);
        % winsorize the data
        if (size(x,1) > size(x,2)+2),  %if 2 more rows than columns - arbitrary
            % winsorize the data
            x = winsorize(x);           
            switch reg_type
                case 'robustfit'
                    % carry out regressions
                    [beta, stats] = robustfit(x,y);
                case  'ridge'
                    % z-score data
                    x = nanzscore(x);
                    [beta] = ridge(y,x,k,0); % no centering and scaling
                    % generate the regression statistics
                    stats = regstats(y,x);
                    stats.t = stats.tstat.t;
                    stats.p = stats.fstat.pval;
                    stats.se = sqrt(diag(stats.covb));
                otherwise
            end
        
            
        % raw model statisics
%         s.X{i,:}    = [ones(size(x,1),1) x];
%         s.t{i,:}    = stats.t;
%         s.p{i,:}    = stats.p;
%         s.b{i,:}    = beta;
%         s.se{i,:}   = stats.se;
        
%             s.X         = [ones(size(x1,1),1) nanzscore(x1)];
            s.X         = [ones(size(x1,1),1) x1];
            s.t         = stats.t; 
            s.p         = stats.p;
            s.b         = beta;
            s.se        = stats.se;
            s.yhat      = s.X * s.b;
            s.y         = ydata(t,:)';
            s.y2         = y;
            s.yhat2      = [ones(size(x,1),1) x1(nidx,:)] * beta;
            s.Mres      = s.y2 - s.yhat2; % fitted model residual (in-sample)
            s.resid     = (s.X * s.b) - ydata(t,:)'; %residual including points not in training set
            s.ybar      = nanmean(s.y2);
            s.SSE       = nansum(s.Mres.*s.Mres);    
            s.SSR       = nansum((s.yhat2 - s.ybar).*(s.yhat2 - s.ybar));
            s.SST       = nansum((s.y2 - s.ybar).*(s.y2 - s.ybar));
            s.dfe       = length(y) - length(s.b);
            s.dfr       = length(s.b) - 1;
            s.dft       = length(y) - 1;
            s.F         = (s.SSR / s.dfr) * (s.SSE / s.dfe);
            s.Fpval      = 1 - fcdf(s.F,s.dfr,s.dfe);
            s.R2         = 1 - s.SSE / s.SST;
            s.adjR2      = 1 - s.SSE / s.SST * (s.dft /s.dfe);
            
            % output statistics etc.
            multi.X(:,:,t)        = s.X;
            multi.beta(t,:)       = s.b;
            multi.se(t,:)         = s.se;
            multi.t(t,:)          = s.t;
            multi.pval(t,:)       = s.p;
            multi.resid(t,:)      = s.resid;
            multi.F(t,:)          = s.F;
            multi.Fpval(t,:)      = s.Fpval;
            multi.R2(t,:)         = s.R2;
            multi.adjrR2(t,:)     = s.adjR2; 
            multi.yhat(t,:)       = s.yhat;
%             multi.yhat2(t,:)      = s.yhat2;
            
            % derived model stats
%             multi.resid(t,:) = y
            
        end;
        
        % derived model statistics
%         s.y{i,:}        = y;
%         s.yhat{i,:}     = s.X{i,:} * s.b{i,:};
%         s.ybar{i,:}     = nanmean(y);
%         s.resid{i,:}    = s.y{i,:} - s.yhat{i,:};
%         s.sse{i,:}      = norm(s.resid{i,:})^2;
%         s.ssr{i,:}      = norm(s.yhat{i,:} - s.ybar{i,:})^2;
%         s.sst{i,:}      = norm(s.y{i,:} - s.ybar{i,:})^2;
%         s.dfe{i,:}      = length(y) - length(s.b{i,:}); % # obs - # beta's
%         s.dfr{i,:}      = length(s.b{i,:}) - 1;
%         s.dft{i,:}      = length(y)-1;
%         s.F{i,:}        = (s.ssr{i,:} / s.dfr{i,:}) * (s.sse{i,:} / s.dfe{i,:});
%         s.pval{i,:}     = 1 - fcdf(s.F{i,:},s.dfr{i,:},s.dfe{i,:});
%         s.R2{i,:}       = 1 - s.sse{i,:} ./ s.sst{i,:};
%         s.adjR2{i,:}    = 1 - s.sse{i,:} ./ s.sst{i,:} * (s.dft{i,:}./s.dfe{i,:}); % for constant term
        % output model statistics

end;