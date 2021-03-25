clc;
clear;
warning off;

% load('Caltech101-7.mat');data = 'Caltech101-7';v = size(X,2);n = size(Y,1);param.dd = [48,40,254,1984,512,928];
load('handwritten.mat');data = 'handwritten';v = size(X,2);n=2000;param.dd = [240,76,216,47,64,6];
% load('MSRC_V1_5views.mat');data = 'MSRC_v1_5v';v = size(X,2);n=210;param.dd = [1302,48,512,256,210];
% load('youtube.mat');data = 'youtube';v = size(X,2);n = 1592;param.dd = [750,750];
for i = 1 :v
    for  j = 1:n
        X{i}(j,:) = ( X{i}(j,:) - mean( X{i}(j,:) ) ) / std( X{i}(j,:) ) ;
    end
end
XX = DataConcatenate(X);
[n,d] = size(XX);
XX = XX';

param.alpha = 1e-0;
param.beta = 1e-0;
param.gamma = 1e-0; 
param.eta = 1e+6; %fixed
selectedFeas = 500;
param.v = v;
param.t = 2;
param.k = 10;
param.n = n;
param.d = d;
param.NITER = 10;

rand('twister',5489);
Ww = constructW_PKN(XX, param.k, 1);
D1 = diag(sum(Ww));
L = D1-Ww;
% p: weight vector; 1*v
% M{v}: clster center; dd(v)*c
% H: shared-label;n*c
% W: feature selection matrix;d*c
[OBJ, P, M, H, W] = MUSFS(X, XX, L,param);

w = [];
for i = 1:d
    w = [w norm(W(i,:),2)];
end
[~,index] = sort(w,'descend');
Xw = XX(index(1:selectedFeas),:);
for m = 1:10
    [y] = kmeans(Xw', param.k);
    result = ClusteringMeasure(Y,y);
    fprintf('dataset = %s, k = %d, ¦Á = %d, ¦Â = %d,¦Ã = %d, ¦Ç = %d, selectedFeas = %d,time = %d,',...
        data,param.k,param.alpha,param.beta,param.gamma,param.eta,selectedFeas,time(j));
    
    fprintf('\n');
    disp(['Best. ACC: ',num2str(result(1))]);
    disp(['Best. NMI: ',num2str(result(2))]);
    disp(['Best. Purity: ',num2str(result(3))]);
    fprintf('\n');
    
    Fin_result(m,(1:3)) = result;
end
result1 = sum(Fin_result);
result2 = result1/10;

fprintf('dataset = %s, k = %d, ¦Á = %d, ¦Â = %d,¦Ã = %d, ¦Ç = %d,selectedFeas = %d , ',...
    data,param.k,param.alpha,param.beta,param.gamma,param.eta,selectedFeas);
fprintf('\n');
disp(['mean. ACC: ',num2str(result2(1))]);
disp(['mean. NMI: ',num2str(result2(2))]);
disp(['mean. Purity: ',num2str(result2(3))]);
fprintf('\n');
    
   