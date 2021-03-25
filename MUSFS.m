function [OBJ,P, M, H, W] = MUSFS(X,XX,L,param)
%% ===================== Parameters =====================
alpha = param.alpha;
beta = param.beta;
gamma = param.gamma;
eta = param.eta;
NITER = param.NITER;
v = param.v;
n = param.n;
d = param.d;
dd = param.dd;
t = param.t;
c = param.c;

%% ===================== initialize =====================
P = 1/v*ones(1,v);
for i = 1:v
    M{i} = rand(dd(i),c);
end
eps = 0.001;
W = rand(d,c);
H = initfcm(c,n);
H = H';

%% ===================== updating =====================

for iter = 1:NITER
     % update M
     for ii = 1:v
         for j = 1:c
             M1 = 0;
             for i = 1:n
                 M1 = M1 + P(ii)*H(i,j)*X{ii}(i,:)';
             end
             M{ii}(:,j)= M1./P(ii)*sum(H(:,j));
         end
     end
     % update W
     sum_W = zeros(n);
     Wi = sqrt(sum(W.*W,2)+eps);
     dd = 0.5./Wi;
     Dw = diag(dd);
     W0 = inv(XX*XX'+gamma/alpha*Dw);
     W = W0*XX*H;
     
     W2 = beta*L+alpha*(eye(n)-XX'*W0*XX);
     for i = 1:v     
        sum_W = sum_W+P(i)*W2;        
     end
     W2 = (sum_W+sum_W')/2;
     SUM_E = zeros(n,c);
     for ii = 1:v
         for i = 1:n
             for j = 1:c
                 distance_1{ii}(i,j) = L2_distance_1(X{ii}(i,:)',M{ii}(:,j));
                 distance_2{ii}(i,j) = distance_1{ii}(i,j)*H(i,j);
                 distance{ii}(i,j) = P(ii)*distance_1{ii}(i,j);
             end
         end
         o(ii) = sum(sum(distance_2{ii}));
         O(ii) = (1/o(ii)).^(1/(t-1));
         SUM_E = SUM_E + sum(distance{ii});
     end
     O1 = sum(O);
    % update p          
     for i = 1:v
       P(i) = O(i)./O1;
     end
     % update H
     H = H.*(eta*H+eps)./(W2*H+SUM_E+eta*H*H'*H+eps);
     H = H*diag(sqrt(1./(diag(H'*H)+eps)));
     
     OBJ(iter) = sum(sum(SUM_E))+alpha*(norm((XX'*W-H),2).^2)+beta*trace(H'*L*H)+gamma*trace(W'*Dw*W);

     if iter == 1
        err = 0;
     else
        err = OBJ(iter)-OBJ(iter-1);
     end
    
    fprintf('iteration =  %d:  obj: %.4f; err: %.4f  \n', ...
        iter, OBJ(iter), err);
    if (abs(err))<1e+0
        if iter > 2
            break;
        end
    end
end










