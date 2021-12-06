%% Remove the border instances
function [ IA ] = BIRCH( X, Y, NumOfNeighbors )

    [idx, ~] = knnsearch(X, X, 'Distance', 'cityblock', 'K', NumOfNeighbors + 1);
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
