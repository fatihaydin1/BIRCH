%% Remove the border instances
function [ IA ] = BIRCH( X, Y, varargin )

    narginchk(2, 3);
    
    if nargin == 2
        param = struct;
        param = setDefaultValues(param);
    elseif nargin == 3
        if ~isstruct(varargin{1})
            error('The param must be a struct');
        end
        param = setDefaultValues(varargin{1});
    end

    [idx, ~] = knnsearch(X, X, 'Distance', param.DistanceMetric, 'K', param.NumOfNeighbors + 1);
    neighbours = Y(idx);
    % Set 1 if neighbors are in different classes, otherwise 0
    nn = (neighbours(:,1) ~= neighbours(:,2:end));
    
    % Different class
    [r, c] = find(nn==1);
    b = [r c];
    [~, ia, ~] = unique(b(:,1));
    B = b(ia, :);
    
    % The same class
    [r, c] = find(nn==0);
    b = [r c];
    [~, ia, ~] = unique(b(:,1));
    A = b(ia, :);
    
    % The difference of two sets
    IA = setdiff(A(:,1), B(:,1));
end



%% Set the parameters to their default values
function [ param ] = setDefaultValues( param )

    field = {'DistanceMetric', 'NumOfNeighbors'};
    TF = isfield(param, field);
    
    if TF(1) == 0
        param.DistanceMetric = 'cityblock';
    end
    
    if TF(2) == 0
        param.NumOfNeighbors = 1;
    end
end
