%% out-of-sample results

% calculate overfitting stats
% caluclate performance of best HP configuration over out-of-sample period


%% load in-sample results 
userpathstr = userpath;
userpathstr = userpathstr(~ismember(userpathstr,';'));
filepath = strcat(userpath,'/Algo_Invest/Algo_Invest_github/preprint_results_to_replicate');
% cd Algo_Invest/workspace/new_Algo

% 1. Form Matrix M = T x N of N different strategies

clear M
for i = 1:4
    load(strcat(filepath,'/Algo_temp', int2str(i)));
    M{1,i} = Algo;
    clear Algo;
end

M = cat(2,M{:});
M(1,:) = 0;
N = size(M,2);
writematrix(M, 'M_matrix_new.txt');

% 2. Partition M into even number S of disjoint sub-matrices M_s

M = M(2:end-4,:); % get convenient even number
U = size(M,1);
S = 16;
M_s = mat2cell(M, U/S*ones(S,1), N);

% 3. Form all combinations C_s of S/2 submatrices M_s
combs = combnk(1:S, S/2);


%% 4. For each combination c of C_s
for i = 1:size(combs,1)
    % 4.1 Form training set J by joining S/2 submatrices M_S in original
    % order
    c = combs(i,:);
    J = M_s(c);
    J = cat(1, J{:});
    % 4.2 Form testing set J_bar as complement of J in M
    c_bar = setdiff(1:S,c);
    J_bar = M_s(c_bar);
    J_bar = cat(1, J_bar{:});
    % 4.3 Form vector R_c of performance statistics on J, and ranks r_c
    R_c = sharpe(J);
    r_c = 1:length(R_c);
    [~,ind] = sort(R_c, 'descend'); 
    r_c(ind) = r_c;    
    % 4.4 Form vector R_bar_c of performance statistics on J_c and ranks
    % r_bar_c
    R_bar_c = sharpe(J_bar);
    r_bar_c = 1:length(R_bar_c);
    [~,ind] = sort(R_bar_c, 'descend'); 
    r_bar_c(ind) = r_bar_c;  
    % 4.5 Select best in-sample strategy
    n_star = find(r_c==min(r_c));
        % IS-OOS pairs for  performance degradation
        IS_OOS(i,:) = [R_c(n_star), R_bar_c(n_star)];
    % 4.6 Define relative rank o_bar_c out-of-sample
    o_bar_c = r_bar_c(n_star)/(N+1);
    % 4.7 Define logit l_c
    l_c(i) = log(o_bar_c/(1-o_bar_c));
end
        
%% 5. Compute the distribution of logits OOS
[f,l] = ecdf(l_c);

PBO = f(max(find(l<=0)));
% PBO is equal to 0.7622

scatter(IS_OOS(:,1), IS_OOS(:,2), 25,'MarkerEdgeColor',[0 .5 .5],...
              'MarkerFaceColor',[0 .7 .7],...
              'LineWidth',1.5);
          
% 6. Compute overfit statistics          
[LS,~,~,~,stats] = regress(IS_OOS(:,2), [ones(length(IS_OOS),1), IS_OOS(:,1)]);
% b0 = 0.6340, b1=-0.9684, R2 = 0.7406



%% DSR

% check clusters
k=2:10; % number of clusters to try
myfunc = @(X,K)(kmeans(X, K)); 
eva = evalclusters(M, myfunc,'silhouette','klist',k);
classes = kmeans(M, eva.OptimalK);
K = eva.OptimalK;

% calculate DSR on training-set
dsr1 = DSR(M(2:end,:), 'std');
% max dsr is 0.9975

% calculate SR on training-set
sr1 = sharpe(M);
% maximum SR is 0.3341

% calculate PSR on test-set
psr1 = PSR(M, 0);
% max psr is 1


%% Performance Curves and Metrics
% Run best performing model over full dataset

% in-sample period
I = ini(2) - 1; % initialisation
% T = updt(2); % training
U = 485;
T = I + U;

% asset returns
R = fts2mat(db_procdata.returns.eret);   %excess
lnRx = fts2mat(db_procdata.returns.ret); %abs
% factor premia
X = fts2mat(db_factors.returns.fmpfts);
% X = fts2mat(db_factors.returns.fmpfts.MKT);

% index (market-cap based)
uni = 'top100';
index1 = fts2mat(db_procdata.indexes.(uni)); 
index1(isnan(index1)) = 0; % NaN to 0
index1 = lagmatrix(index1,1);
index1(1,:) = 0;
% returns index
index2 = double(~isnan(R));
% investible assets at t+1  = {index at t} and {returns at t+1}
index = double(and(logical(index1), logical(index2)));

% no. assets
n = size(R,2);
N = sum(index~=0,2);

% characteristics data
Z = NaN*ones(length(R), n, 7); % allocate memory
% create data matrix
for i = 1:size(db_procdata.char_data,2)
   % create matrix and apply index
   Z(:,:,i) = index.*fts2mat(db_procdata.char_data{1,i});
end

% characteristic model definition
ch_mdl{1} = [4 2];         % bvtp & mv
ch_mdl{2} = [4 2 6];       % bvtp & mv & moml
ch_mdl{3} = [4 2 7];       % bvtp & mv & moms
ch_mdl{4} = [4 2 6 7];     % bvtp & mv & moml & moms
% number of models
m = length(ch_mdl);
% select model
j = 2;


%% Remove missing columns

check = nansum(index);
col_ind = (check==0);
index(:, col_ind) = [];
Z(:, col_ind, :) = [];
R(:, col_ind) = [];
lnRx(:, col_ind) = [];

% no. assets
n = size(R,2);
N = sum(index~=0,2);

clear check;

%% 1. Choose best performer
best = find(sr1 == max(sr1)); % index in M
mdl = ceil(best(1)/2700); % best mdl is 2
ind = mod(best(1),2700); % best HP is 657
HP_set = dlmread('HP_set.txt');
HP_set = mat2cell(HP_set, ones(2700,1), 6);


% 2. run best performer on full dataset Run on server
% Out_Algo1 = AlgoInvest(I, T, R, lnRx, X, Z, index, ch_mdl{mdl}, HP_set{ind});

% 3. Load Best Performer - Prepared on Xeon
%{
userpathstr = userpath;
userpathstr = userpathstr(~ismember(userpathstr,';'));
filepath = strcat(userpath,'/Algo_Invest/workspace/portfolios');
load('/home/andrew/Dropbox/MATLAB/Algo_Invest/workspace/new_Algo/Out_Algo.mat');
%}

% 2. In-sample performance
% In_Algo = exp(cumsum(M(:,best(1)),1));

OSsr_Algo = sharpe(Out_Algo1.logR_w(488:972)); % 0.0562
ISsr_Algo = sharpe(Out_Algo1.logR_w(2:487)); % 0.3309

OSsr_nd = sharpe(BM_logR(I+U+1:end-1,1)); % 0.1007
ISsr_nd = sharpe(BM_logR(I:I+U,1)); % 0.2001

OSsr_cap = sharpe(BM_logR(I+U+1:end-1,2)); % 0.0480
ISsr_cap = sharpe(BM_logR(I:I+U,2)); % 0.1182

% 3. Probabilistic Sharpe Ratio
Out_Algo = Out_Algo1.logR_w;
OSpsr_Algo = PSR(Out_Algo(488:972),0); % 0.8927
ISpsr_Algo = PSR(Out_Algo(2:487),0); % 1



fts_BM = db_factors.returns.indpft;
BM_logR = fts2mat(fts_BM);

OSpsr_nd = PSR(BM_logR(I+U+1:end-1,1),0);
ISpsr_nd = PSR(BM_logR(I:I+U,1),0);

OSpsr_cap = PSR(BM_logR(I+U+1:end-1,2),0);
ISpsr_cap = PSR(BM_logR(I:I+U,2),0);


% 4. turnover
AlgoTO = turnover2(Out_Algo1.w, lnRx(243:1214,:));

In_AlgoTO = AlgoTO(1:487);
In_ndTO = db_factors.returns.ndTO(243:729);
In_capTO = db_factors.returns.capTO(243:729);

Out_AlgoTO = AlgoTO(488:972);
Out_ndTO = db_factors.returns.ndTO(730:1214);
Out_capTO = db_factors.returns.capTO(730:1214);

In_TO = [mean(In_AlgoTO) mean(fts2mat(In_ndTO)) mean(fts2mat(In_capTO))]; % [0.7165, 0.1247, 0.0763]
Out_TO = [mean(Out_AlgoTO) mean(fts2mat(Out_ndTO)) mean(fts2mat(Out_capTO))]; % [0.5680, 0.0630, 0.0504]
%% Out-of-sample results to python


% create fints from matrix
dates = db_procdata.returns.ret.dates;
freq = db_procdata.returns.ret.freq;
dates = dates(end-972:end-1,:);

R_p = Out_Algo1.logR_w;
R_g = Out_Algo1.logR_g;
R_s = Out_Algo1.logR_s;
R_a = Out_Algo1.logR_a;

% algo returns
fts_out_algo  = fints(dates, [R_p, R_g, R_s, R_a], {'Algo', 'GMV', 'Sys', 'Act'}, freq, 'Returns');
fts2ascii(fullfile(filepath,strcat('FTS_OUT_ALGO','.txt')), fts_out_algo);

% benchmark returns
fts_benchmark = db_factors.returns.indpft;
fts2ascii(fullfile(filepath,strcat('FTS_BENCHMARK','.txt')), fts_benchmark);


% fmpfts = fints(db_procdata.returns.ret.dates,[rHML, rSMB, rMKT],{'HML','SMB','MKT'},db_procdata.returns.ret.freq,'Returns');