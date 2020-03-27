% Preprint Empirical Data Generation

%% 1. Load Data
% prep data in Data_prep_002aa

clear;clc;
warning off;

% load data
load('procdata1_w');

% indexes
indexes     = {'top40', 'top100', 'top160'};
nindex      = [40 100 160];

% choose index
% for ii = 1:numel(indexes)
% uni=indexes{ii};
uni = 'top100';
findex = db_procdata.indexes.(uni);
index = double(fts2mat(db_procdata.indexes.(uni)));
index(index==0) = NaN;

%% split data into initialization and update
% end of in-sample period


insample = datenum('28-Dec-2007');
insample = find(findex.dates==insample);

% split into initialisation and update (one-third / two-third)
ini = [1, round(insample/3) ];
updt = [round(insample/3) + 1, insample];



%% Sort companies by characteristics into quantiles
% index has not been lagged (findex has)
% characteristics have been lagged by 4 weeks

% we lag variables from published financial records.
clear nb nm nm3;
% create temp local characteristic variables
mv = lagts(db_procdata.char_data{2},4); %mv
bvtp = lagts(db_procdata.char_data{4},4); %bvtp

% quintile sorts - value
nb = ntile(transpose((index .* fts2mat(bvtp))),3,'descend'); 
db_quintile.sorts.nb = fints(bvtp.dates,transpose(nb),fieldnames(bvtp,1),bvtp.freq,bvtp.desc);
% quintile sort - size - 2 quintiles
nm = ntile(transpose(index .* fts2mat(mv)),2,'descend'); 
db_quintile.sorts.nm = fints(mv.dates,transpose(nm),fieldnames(mv,1),mv.freq,mv.desc);
% quintile sort - size - 3 quintiles
nm3 = ntile(transpose(index .* fts2mat(mv)),3,'descend');
db_quintile.sorts.nm3 = fints(mv.dates,transpose(nm3),fieldnames(mv,1),mv.freq,mv.desc);
% delete temp variables
clear mv bvtp;
clear nb nm nm3; 


%% Create Fama-French Factor portfolios
% lag portfolios to apply return
% eg. at time t, intersection portfolio return: HS(t-1)*ret(t)

nbx = fts2mat(db_quintile.sorts.nb);
nmx = fts2mat(db_quintile.sorts.nm);
% nm3x = fts2mat(db_quintile.sorts.nm3);
% create the index for the different bins
small  = (nmx==2);
big    = (nmx==1);
high   = (nbx==1);
medium = (nbx==2);
low    = (nbx==3);
% create LS (low small) intersection of nbvtp and nmv equally weighted
db_quintile.intersection.LS = low & small;
db_quintile.intersection.MS = medium & small;
db_quintile.intersection.HS = high & small;
db_quintile.intersection.LB = low & big;
db_quintile.intersection.MB = medium & big;
db_quintile.intersection.HB = high & big;

% Compute the normalizations
db_quintile.norms.normLS = transpose(sum(transpose(db_quintile.intersection.LS)));
db_quintile.norms.normMS = transpose(sum(transpose(db_quintile.intersection.MS)));
db_quintile.norms.normHS = transpose(sum(transpose(db_quintile.intersection.HS)));
db_quintile.norms.normLB = transpose(sum(transpose(db_quintile.intersection.LB)));
db_quintile.norms.normMB = transpose(sum(transpose(db_quintile.intersection.MB)));
db_quintile.norms.normHB = transpose(sum(transpose(db_quintile.intersection.HB)));

% clear local variables
clear nbx nmx nm3x small big high medium low;
 
%% Create the indices for the intersection portfolios
% retrieve stock returns
ret = db_procdata.returns.ret;
retx = fts2mat(db_procdata.returns.ret);

%create temp vars
tmp_inputs1 = {'LS', 'MS', 'HS', 'LB', 'MB', 'HB'};
[LS, MS, HS, LB, MB, HB] = create_vars(db_quintile.intersection, tmp_inputs1);
tmp_inputs2 = {'normLS', 'normMS', 'normHS', 'normLB', 'normMB', 'normHB'};
[normLS, normMS, normHS, normLB, normMB, normHB] = create_vars(db_quintile.norms, tmp_inputs2);

% compute the returns - equal  weighting
db_quintile.intersection.rLS = transpose(nansum(transpose(retx .* LS))) ./ normLS; 
db_quintile.intersection.rMS = transpose(nansum(transpose(retx .* MS))) ./ normMS;
db_quintile.intersection.rHS = transpose(nansum(transpose(retx .* HS))) ./ normHS;
db_quintile.intersection.rLB = transpose(nansum(transpose(retx .* LB))) ./ normLB;
db_quintile.intersection.rMB = transpose(nansum(transpose(retx .* MB))) ./ normMB;
db_quintile.intersection.rHB = transpose(nansum(transpose(retx .* HB))) ./ normHB;
form_returns = [db_quintile.intersection.rLS, db_quintile.intersection.rMS, ...
    db_quintile.intersection.rHS, db_quintile.intersection.rLB, ... 
    db_quintile.intersection.rMB, db_quintile.intersection.rHB];
db_quintile.intersection.retfts = fints(ret.dates,form_returns,{'LS','MS','HS','LB','MB','HB'},ret.freq,'Returns');

% clear temp variables
clear retx form_returns ret; 
clear tmp_inputs1 LS MS HS LB MB HB;
clear tmp_inputs2 normLS normMS normHS normLB normMB normHB;


%% Create Fama-French Factor Replicating Portfolios

% local temp variables
tmp_inputs = {'rHB','rHS','rMB','rMS','rLB','rLS'};
[rHB, rHS, rMB, rMS, rLB, rLS] = create_vars(db_quintile.intersection, tmp_inputs);
% Create the quarterly sorts of H(igh)-M(inus)-L(ow) Book-to-Market
rHML = (1/2)*((rHB + rHS) - (rLB + rLS));
% Create the quarterly sorts of S(mall)-M(inus)-B(ig) on Log-size
rSMB = (1/3)*((rHS + rMS + rLS) - (rHB + rMB +rLB));
% Create a market index (market cap weighted - balanced portfolio)
mvx = exp(index .* fts2mat(lagts(db_procdata.char_data{2})));
retx = fts2mat(db_procdata.returns.ret);;
normMV = transpose(nansum(transpose(mvx)));
% normND = min(nansum(index,2), nansum(~isnan(retx),2)); % min of index or populated returns
normND = nansum(~isnan(index.*retx),2);

% rMKT = (transpose(nansum(transpose(retx .* mvx))) ./ normMV) - fts2mat(db_procdata.returns.rfr);
rMKT = (transpose(nansum(transpose(retx .* mvx))) ./ normMV) - fts2mat(db_procdata.returns.rfr);
% naive diversification
rIND = nansum(index.*retx,2)./normND;
% naive weights
ndw = index ./ normND;
% naive turonver
ndTO = turnover(ndw, retx);
% net return assuming 50 basis point proportional transaction costs
c = 0.005;
netIND = rIND + log(1-c*ndTO);

% cap weighted
rCAP = (transpose(nansum(transpose(retx .* mvx))) ./ normMV);
% cap weights
capw = mvx ./ normMV;
% cap weights turnover - will be zero if we consider whole market, but
% there are rebalancing costs associated with changeRemove Series without all datapoints in the index
% constituents
capTO = turnover(capw, retx);
% net return
netCAP = rCAP + log(1-c*capTO);

% store returns struct
db_factors.returns.rHML = rHML;
db_factors.returns.rSMB = rSMB;
db_factors.returns.rMKT = rMKT;
db_factors.returns.rIND = rIND;
db_factors.returns.rCAP = rCAP;
db_factors.returns.rIND = rIND;
db_factors.returns.netIND = netIND;
db_factors.returns.netCAP = netCAP;

% clear vars
clear mvx retx normMV;

% risk-free rate
rfr = fts2mat(db_procdata.returns.rfr);

% Store factor returns in separate fints structure
fmpfts = fints(db_procdata.returns.ret.dates,[rHML, rSMB, rMKT],{'HML','SMB','MKT'},db_procdata.returns.ret.freq,'Returns');
indpft = fints(db_procdata.returns.ret.dates,[rIND,rCAP, netIND, netCAP, rfr ],{'ND','CAP', 'netND', 'netCAP', 'rfr'},db_procdata.returns.ret.freq,'Benchmark Portfolio');
prcfmp = fillts(fmpfts(12:end),0);
prcfmp(1) = 0;
prcfmp = exp(cumsum(prcfmp));

prcind = fillts(indpft(12:end),0);
prcind(1) = 0;
prcind = exp(cumsum(prcind));

% store vars
db_factors.returns.fmpfts = fmpfts;
db_factors.returns.prcfmp = prcfmp;
db_factors.returns.indpft = indpft;
db_factors.returns.prcind = prcind;
% store turnover
db_factors.returns.ndTO =  fints(db_procdata.returns.ret.dates, ndTO ,{'ND'},db_procdata.returns.ret.freq,'Turnover');
db_factors.returns.capTO = fints(db_procdata.returns.ret.dates, capTO ,{'CAP'},db_procdata.returns.ret.freq,'Turnover');

%{
newpath = '/home/andrew/Dropbox/MATLAB';
userpath(newpath);
userpathstr = userpath;
userpathstr = userpathstr(~ismember(userpathstr,';'));
filepath = fullfile(userpathstr,'Algo_Invest/workspace/portfolios_new');
fts2ascii(fullfile(filepath, 'Benchmark_JSE_logR.txt' ), db_factors.returns.indpft);
%}

%clear variables
clear tmp_inputs rHB rHS rMB rMS rLB rLS rHML rSMB rMKT fmpfts prcfmp;
clear capTO capw ndTO ndw netCAP netIND normND rCAP rfr rIND prcind indpft;

clc;