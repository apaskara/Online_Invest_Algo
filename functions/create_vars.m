function [varargout] = create_vars(struct_path, fields)
% function returns variables from specified fields
% length of fields needs to match length of output vector

for i=1:length(fields),
    varargout{i} = getfield(struct_path, fields{i});
end;