function [invA] = naninv_001ac(A, varargin)
%
% version: 001ac - change varargin to rather be inverse type
%
% removes nans from matrix and takes inverse, and produces matrix with NaNs
% this is only used for matrix which includes stocks THAT ARE NOT
% INVESTABLE. ie stocks which are not part of the market
% In this case, the NaN does not represent missing data.
%
% if the NaN in the came from illiquid stocks, it would be best to impute
% the missing data rather than remove.
%
% Note the following required properties of the input matrix:
%   1. symmetric
%   2. if A[ij] = NaN, then either A[i.] = NaN or A[.j]=NaN

    %
    % inverse type default
    type = 'inv';
    % optional inputs
    if numel(varargin) > 0
        type = varargin{1};
    end
    % find size of matrix
    [rn,cn] = size(A);
    % find nans in the matrix
    nanind = (sum(~isnan(A))>0);
    % remove nans from matrix - will be whole columns and rows (see 2. above)
    A1 = A(nanind',nanind);

    % fill invA with NaNs
    invA = NaN*ones(rn,cn);
    % if matrix is not empty
    if ~isempty(A1)

        % clean matrix
        switch type
            case 'inv'
                tmp_invA = inv(A1);

            case 'pinv'
                tmp_invA = pinv(A1);   
        end
        % fill invA elements
        invA(nanind',nanind)=tmp_invA;
    end % if

end % function



