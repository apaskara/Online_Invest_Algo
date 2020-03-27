function [Algo] = AlgoInvest_001aa(I,T,R,lnRx,X,Z, index, ch_mdl, varargin)

% Inputs
% ------------------
% I         - initialisation length
% T         - total length
% R         - excess returns data
% Rx        - log absolute returns
% X         - factor returns
% Z         - characteristics data
% index     - investible assets
% ch_mdl    - char var index

% TBDL
% change inputs to R and rfr


%% Hyper-parameters

% k_s       - systematic returns shrinkage intensity
% k_a       - active returns shrinkage intensity
% lambda_s  - systematic model memory factor (larger implies smoother forecast err)
% lambda_a  - active model memory factor
% g_s       - risk aversion systematic
% g_a       - risk aversion active 

% defaults {k_s k_a lambda_s lambda_a g_s g_a}
optargs = {0.02 0.1 0.98 0.9 4 5};

% change to cell
varargin = num2cell(varargin{:});

% skip optional inputs if they are empty
newVals = cellfun(@(x) ~isempty(x), varargin);

% replace defaults
optargs(newVals) = varargin(newVals);

% place in variables
[k_s, k_a, lambda_s, lambda_a, g_s, g_a] = optargs{:};



%% Initialise

% clear vars
clear b_0 invR0 Sigma_f0 mu_f0 Omega_a0 e_a0 Omega_pi0 e_pi0 Omega_bm0 e_bm0 d_0 D D_0 mu_0 pi_0;

% no. assets
n = size(R,2);
N = sum(index~=0,2);

% memory 
lambda_s0 = 0.98; % sys
lambda_a0 = 0.9; % act

% systematic return betas
B_0 = online_ts_reg(X(1:I,:), R(1:I,:), 'y', {'lambda', lambda_s0}, {'weight', 'std'});
b_0 = squeeze(B_0.beta(end,:,:));
invR0 = B_0.invR0;

% factor covariance and premium
[Sigma_f0, ~, mu_fI] = ewma_cov(lambda_s0, X(1:I,:), {'mu0', nanmean(X)}, {'COV_0', nancov(X)}); 
mu_f0 = mu_fI(end,:);


% characteristic model deltas
lag = 4;
D_0 = cs_reg(Z(1:I,:,ch_mdl) , index(1:I,:).*R(1:I,:), 'robustfit', lag);
[d_0,~] = ewma_mean(lambda_a0, D_0.beta);


% cross-sectional model return uncertainty
% forecast
for t=lag+1:I
    % predictor data
    z = [ones(n,1) squeeze(Z(t-lag+1,:,ch_mdl))];
    z(isnan(z)) = 0;
    % prediction
    mu_0(t,:) = (z*d_0(t-1,:)')';
end
% prediction uncertainty
err_a0 = R(1:I,:) - mu_0; % errors￼
[Omega_a0, ~, e_a0] = ewma_cov(lambda_a0, err_a0, {'mu0', nanmean(err_a0)}, {'COV_0', nancov(err_a0)});
e_a0 = e_a0(end,:);
d_0 = d_0(end,:);



% total model return estimation uncertainty - CHANGE
% [Omega_a0, ~, e_a0] = ewma_cov(lambda_s, B_0.fc_err, {'mu0', nanmean(B_0.fc_err)}, {'COV_0', nancov(B_0.fc_err)});
% e_a0 = e_a0(end,:);

% systematic forecast prediction uncertainty
for t=1:I
    pi_0(t,:) = (squeeze(B_0.beta(t,:,2:end))*mu_fI(t,:)')'; % estimates
end
pi = pi_0(end,:); % initialise forecast
pi_0 = lagmatrix(pi_0,1); % lag estimates
err_pi0 = R(1:I,:) - pi_0; % errors
[Omega_pi0, ~, e_pi0] = ewma_cov(lambda_s0, err_pi0, {'mu0', nanmean(err_pi0)}, {'COV_0', nancov(err_pi0)});
e_pi0 = e_pi0(end,:);


% benchmark prediction uncertainty
w_bm = 1/N(I,:)*ones(n,1); % 1/n 
k = 0.01; % shrinkage intensity
Sigma_star = eye(n); % shrinkage target
Sigma_0 = k*Sigma_star + (1-k)*(b_0(:,2:end)*Sigma_f0*b_0(:,2:end)' + diag(diag(Omega_a0))); % cov matrix
Sigma_0(isnan(Sigma_0)) = 0; % nan management
pi_bm = Sigma_0*w_bm; % benchmark estimates
% pi_bm = (IDX_0.*Sigma_0)*w_bm; % benchmark estimates
err_bm0 = repmat(pi_bm',[I 1]) - R(1:I,:);  % forecast errors - HACKED
[Omega_bm0, ~, e_bm0] = ewma_cov(lambda_s0, err_bm0, {'mu0', nanmean(err_bm0)}, {'COV_0', nancov(err_bm0)});
e_bm0 = e_bm0(end,:);



%% One-step Update
clear B_loop pi w_g w_s w;
% clear e_bm0 err_bm0 e_bm err_bm;
% number of periods (update)
U = T - I;

% allocate memory
B_loop = cell(U,1); % beta update
pi = NaN*ones(U,n); % Sys E[R]
mu = NaN*ones(U,n); % Char model
a = NaN*ones(U,n); % alpha
a_BL = NaN*ones(U,n); % alpha
w_g = NaN*ones(U,n); % gmv
w_s = NaN*ones(U,n); % sys
w_a = NaN*ones(U,n); % act
w = NaN*ones(U,n); % comb
condno_g = NaN*ones(U,1);
condno_s = NaN*ones(U,1);
condno_a = NaN*ones(U,1);
condno = NaN*ones(U,3);
% g_s = NaN*ones(U,1);
% g_a = NaN*ones(U,1);
% g = NaN*ones(U,2);
err_pi = NaN*ones(U,n); % sys
err_a = NaN*ones(U,n); % combined
err_bl = NaN*ones(U,n); % BL 
err_bm = NaN*ones(U,n); % benchmark


Delta = NaN*ones(U,size(d_0,2));
D = NaN*ones(U,size(d_0,2));



tic
for t = 1:U
    % factor betas
    B_loop{t} = online_ts_reg(X(I+t,:), R(I+t,:), 'y', {'lambda', lambda_s}, {'weight', 'huber'}, {'invR0', invR0}, {'B0', b_0'});
    b_t = squeeze(B_loop{t}.beta(:,:,:));
    % factor covariance and premium
    [Sigma_f,~, mu_f] = ewma_cov(lambda_s, X(I+t,:), {'mu0', mu_f0}, {'COV_0', Sigma_f0});      
    % systematic model return prediction uncertainty
    err_pi(t,:) = R(I+t,:)-pi(t,:);
    [Omega_pi, ~, e_pi] = ewma_cov(lambda_s, err_pi(t,:), {'mu0',e_pi0}, {'COV_0', Omega_pi0});
    % benchmark uncertainty
    err_bm(t,:) = R(I+t,:)-pi_bm';
    [Omega_bm, ~, e_bm] = ewma_cov(lambda_s, err_bm(t,:), {'mu0', e_bm0}, {'COV_0', Omega_bm0});
    % deltas - characteristic payoffs
    delta = cs_reg(Z(I+t-lag,:,ch_mdl) , index(I+t,:).*R(I+t,:) , 'robustfit', lag); % cross-sectional regression
    Delta(t,:) = delta.beta;
    [~,d_1] = ewma_mean(lambda_a, Delta(t,:), d_0); % smooth payoff
    D(t,:) = d_1;
    % alpha model return prediction uncertainty
    err_a(t,:) = R(I+t,:)-mu(t,:);
    [Omega_a, ~, e_a] = ewma_cov(lambda_a, err_a(t,:), {'mu0', e_a0}, {'COV_0', Omega_a0});
    
    
    % idiosyncratic risk
    Sigma_e0 = zeros(n);
    
    % expected systematic returns prediction 
    % forecast factor return
    f_hat = mu_f;
    % forecast factor beta
    b_t1 = b_t(:,2:end); % flat forecast, exclude intercept
    % forecast systematic returns (no alpha)
    pi(t+1,:) = (b_t1*f_hat')';
    
    % active return prediction
    % predictor data
    z = [ones(n,1) squeeze(Z(I+t-lag+1,:,ch_mdl))];
    z(isnan(z)) = 0;
    % prediction
    mu(t+1,:) = (z*d_1')';
    a(t+1,:) = mu(t+1,:) - pi(t+1,:); 
    

    
    % covariance matrices
    Sigma_t1 = b_t1*Sigma_f*b_t1' + Sigma_e0; % conditional Cov
    Sigma_star = eye(n); % shrinkage target
    Sigma_gt1 = k_s*Sigma_star + (1-k_s)*(Sigma_t1 + diag(diag(Omega_a))); %GMV 
    Sigma_gt1(isnan(Sigma_gt1)) = 0; % nan management
%     Sigma_pit1 = k*Sigma_star + (1-k)*(Sigma_t1 + diag(diag(Omega_pi)));    % Sys
%     Sigma_at1 = k*Sigma_star + (1-k)*(Sigma_t1 + diag(diag(Omega_a)));  % Act
    
    % If want to stabilise systematic returns

%{
    % Reverse Optimise benchmark
    w_bm = 1/N(I+t)*ones(n,1);
    pi_bm = Sigma_gt1*w_bm;
    % mixed strategic estimates 
    [pi_m, Omega_pim] = MixEst(pi(t+1,:)', pi_bm, Omega_pi, Omega_bm);
%     [pi_m, Omega_pim] = MixEst(pi(t+1,:)', pi(t+1,:)', Omega_pi, Omega_pi);  % no mixing 
%}    
    
    % No stabilising of systematic returns
    pi_m = pi(t+1,:)';
    Omega_pim = Omega_pi;

    % mixed alpha estimates
    [a_BL(t+1,:), Omega_am] = MixEst(zeros(n,1), a(t+1,:)', diag(diag(Omega_pim)), diag(diag(Omega_a)));
    a_bl = a_BL(t+1,:)';
    
    % Covariance matrices
    Sigma_pit1 = k_s*Sigma_star + (1-k_s)*(Sigma_t1 + diag(diag(Omega_pim)));    % Sys
    Sigma_at1 = k_a*Sigma_star + (1-k_a)*(Sigma_t1 + diag(diag(Omega_am)));  % Act

    % filter investible set
    idx_t1 = index(I+t+1,:); % vector filter
    IDX_t1 = idx_t1'*idx_t1; % matrix filter
    idx_t1(idx_t1==0) = NaN;
    IDX_t1(IDX_t1==0) = NaN;
    

    [w_g(t+1,:),~]  = GMV(IDX_t1.*Sigma_gt1, 'pinv'); %GMV
    [w_s(t+1,:),~,~] = STRAT(IDX_t1.*Sigma_pit1, idx_t1'.*pi_m, g_s, 'inv');  %Sys
    [w_a(t+1,:),~,~]  = STRAT(IDX_t1.*Sigma_at1, idx_t1'.*a_bl, g_a,  'inv'); %Act - note that strat and act have same solution
    
    w(t+1,:) = w_g(t+1,:)+w_s(t+1,:)+w_a(t+1,:); %  combined
    
    
    % re-initialise
    b_0 = squeeze(B_loop{t}.beta(end,:,:)); % factor betas
    invR0 = B_loop{t}.invR0; % R matrix (search direction?)
    Sigma_f0 = Sigma_f; % factor covariance
    mu_f0 = mu_f; % factor premium
    Omega_a0 = Omega_a; % full model forecast uncertainty
    e_a0 = e_a; % full model mean forecast error
    Omega_pi0 = Omega_pi; % Systematic forecast uncertainty
    e_pi0 = e_pi; % systematic model mean forecsat error
    Omega_bm0 = Omega_bm; % Benchmark uncertainty
    e_bm0 = e_bm; % benchmark model mean forecast error
    
end
time = toc;

%% calculate returns

% convert log returns to returns
% returns - Need to check log-returns \ returns issue
% lnRx = fts2mat(db_procdata.returns.ret);
% lnRx(:,col_ind) = [];
Rx = exp(lnRx);
Rx = Rx(I+1:T,:); % training period

% asset universe
index(index==0)=NaN;

% Algorithm
Algo.w_g = w_g(1:end-1,:);
Algo.w_s = w_s(1:end-1,:);
Algo.w_a = w_a(1:end-1,:)./nansum(abs(w_a(1:end-1,:)),2); % leverage constraint

Algo.w = Algo.w_g + Algo.w_s + Algo.w_a;


% these are log-returns
Algo.logR_g = log(nansum(Algo.w_g.*Rx,2));
Algo.logR_s = log(1+(nansum(Algo.w_s.*Rx,2))); % '1+' since tactical portfolio
Algo.logR_a = log(1+(nansum(Algo.w_a.*Rx,2)));
Algo.logR_w = log(nansum(Algo.w.*Rx,2));





