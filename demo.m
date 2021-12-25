%% Demos for the BIRCH algorithm
clc;
clear;

% Select a data set
dataset = load('electricity');
fns = fieldnames(dataset);
[ X, Y ] = divideTable( dataset.(fns{1}) );

param.DistanceMetric = 'euclidean';
param.NumOfNeighbors = 1;

ACC = demo1(X, Y, 'KNN', param);

[ idx, newX, newY, R, T ] = demo2( X, Y, param );

[ idx2, newX2, newY2, R2, T2 ] = demo3( X, Y );

clear dataset;
clear fns;



%%
function [ ACC ] = demo1( X, Y, classifier, param )

    predictions = repmat(Y, 1, 2);
    indices = crossvalind('Kfold', Y, 10);
        
    for i = 1:10
        fprintf('%d',i);
        test = (indices == i);
        train = ~test;
                
        trainY = Y(train,:);
        trainX = X(train,:);
        testX = X(test,:);
        
        idx = BIRCH(trainX, trainY, param);
        newTrainX = trainX(idx, :);
        newTrainY = trainY(idx);

        switch classifier
            case 'KNN'
                Mdl = fitcknn(newTrainX, newTrainY, 'NumNeighbors', 1);
            case 'CART'
                Mdl = fitctree(newTrainX, newTrainY);
                predictions(test, 2) = predict(Mdl, testX);
            case 'NB'
                % "normal", "mn", "kernel", "mvmn".
                Mdl = fitcnb(newTrainX, newTrainY, 'DistributionNames', 'normal');
                predictions(test, 2) = predict(Mdl, testX);
            case 'SVM'
                % "linear", "gaussian", "rbf", "polynomial"
                t = templateSVM('Standardize', true, 'KernelFunction', 'linear');
                Mdl = fitcecoc(newTrainX, newTrainY, 'Learners', t);
                predictions(test, 2) = predict(Mdl, testX);
        end
        predictions(test, 2) = predict(Mdl, testX);
    end
    ACC = sum(predictions(:,1) == predictions(:,2))*100/length(Y);
end



%%
function [ idx, newX, newY, R, T ] = demo2( X, Y, param )

    m = numel(Y);
    tic;
    idx = BIRCH(X, Y, param);
    T = toc; % Running time
    newX = X(idx, :);
    newY = Y(idx);
    % Calculate the reduction rate
    R = (m - numel(idx))*100/m;
end



%%
function [ idx, newX, newY, R, T ] = demo3( X, Y )

    m = numel(Y);
    tic;
    idx = BIRCH(X, Y);
    T = toc;
    newX = X(idx, :);
    newY = Y(idx);
    % Calculate the reduction rate
    R = (m - numel(idx))*100/m;
end



%% Separate the dataset into the input matrix and the output vector
function [ X, Y ] = divideTable( DATASET )

    if istable(DATASET)
        X = table2array(DATASET(:,1:end-1));        
        Y = categorical(DATASET.Class);
    else
        error('The parameter must be a table, not a %s.', class(DATASET));
    end
end
