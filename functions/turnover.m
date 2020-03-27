function [turnover] = turnover(weights, ret)
% c is proportional transaction costs - default is 50 basis points
% need to set first observation ot zero for weights


% weights at end of period before reblancing (br)
weights_br  = weights.*exp(ret);
% lag weights 1 period
weights_br = lagmatrix(weights_br,1);
% set nan weights to zero
weights_br(isnan(weights_br)) = 0;
% weights(isnan(weights)) = 0;

% calculate turnover
turnover = nansum(abs(weights_br-weights),2);

% series name
% if isempty(varargin)
%     name = 'ptf';
% else
%     name = varargin{1};
% end
% 
% turnover = fints(weights.dates, turnover, name, weights.freq, 'turnover');


