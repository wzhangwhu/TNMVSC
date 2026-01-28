%% % This demo performs multi-view clustering using TNMVSC algorithm
clc
clear

%%
currentFolder = pwd;
addpath(genpath(currentFolder))


 %% example data
  dataset={'MSRCV1'}   %sample*feature

%%
for ii = 1:length(dataset)
    ii
    load(dataset{ii})
    
    %% multi-view
    V = length(X);%本程序数据X维度:M*N; feature*sample。图像数据:sample*feature

    for i = 1:V
        X{i} = X{i}';
    end
    true_label = Y;
    
    %%
    nCluster = length(unique(true_label));

    lambda1 = 10;%10
    lambda2 = 10;%10
    maxIter = 30;
    tic
    
    %% Initialization S by KNN
    options = [];
    options.NeighborMode = 'KNN';
    options.k = 10;
    options.WeightMode = 'Binary'; % Cosine Binary HeatKernel

    for v = 1:V
        S{v} = constructW(X{v}',options);
    end
    
    %% main function
    [Zn, error] = TNMVSC_opt(X, S, lambda1, lambda2, maxIter);
    %     plot(error)
    M = retain(Zn);
    W = postprocessor(M);
    result_label = new_spectral_clustering(W,nCluster);
    %     result = [ACC NMI Fscore Precision ARI Purity Recall Entropy];
    result = Clustering8Measure(true_label, result_label);
    
end
save result.mat result
result
