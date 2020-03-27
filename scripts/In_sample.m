
% Perpare in-sample results


%% Offline choices 

% in-sample period
I = ini(2)-1; % initialisation
% T = updt(2); % training
U = 485;
T = I + U;

% asset returns
R = fts2mat(db_procdata.returns.eret);
lnRx = fts2mat(db_procdata.returns.ret);
% factor premia
X = fts2mat(db_factors.returns.fmpfts);


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

%% Hyper-parameter set and Performance metric

clear Algo

k_s = 0.01:0.02:0.1;            % strategic covariance shrinkage
k_a = 0.1:0.1:0.3;             % active covariance shrinkage
lambda_s = 0.95:0.01:0.98;     % memory systematic return 
lambda_a = 0.9:0.01:0.94;       % memory active return 
g_s = 3:5;                        % risk tolerance strategic
g_a = 4:6;                        % risk tolerance active


[Ks, Ka, Ls, La, Gs, Ga] = ndgrid(k_s, k_a, lambda_s, lambda_a, g_s, g_a);

% initialise 
Algo = cell(numel(Ks),1);   % Algo performance
HP_set = cell(numel(Ks),1); % hyper-parameter set

% test
j = 1; i = 1;
HP_set{i} = [Ks(i), Ka(i), Ls(i), La(i), Gs(i), Ga(i)];
Algo{i} = AlgoInvest(I, T, R, lnRx, X, Z, index, ch_mdl{j}, HP_set{i});

%{
% for each model
for j = 1:length(ch_mdl)
    % for each HP configuratiom
    for i = 1:numel(Ks)
        tic
        HP_set{i} = [Ks(i), Ka(i), Ls(i), La(i), Gs(i), Ga(i)];
        Algo{i} = AlgoInvest(I, T, R, lnRx, X, Z, index, ch_mdl{j}, HP_set{i});
        time = toc;
    end
    % write files to disk
    save(strcat('Algo_struct',int2str(j)), Algo);
    Algo_ret = Algo{j}.logR_w;
    save(strcat('Algo_ret', int2str(j)), 'Algo_ret') ;
    clear Algo Algo_ret
end
%}

% save HP set
writecell(HP_set, 'HP_set.txt');


