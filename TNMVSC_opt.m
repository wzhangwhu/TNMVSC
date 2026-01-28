function [Zn, error] = TNMVSC_opt(X, S, lambda1, lambda2, maxIter)
%% function TSMVSC_opt
% normalize data to lie in [-1 1]
V = length(X);
for i = 1:size(X,2)
    X{i} = X{i}./(repmat(sqrt(sum(X{i}.^2,1)),size(X{i},1),1)+eps);
end
[~, N] = size(X{1});

for v = 1:V
    D{v} = diag(sum(S{v}'));
    LL{v} = D{v} - S{v};
end

%% ADMM parameters
rho = 2.7;%2.8
alfa = 1.2;
theta = 1;
sX = [N, N, V];
miu = 0.01;
miu_max = 1e8;

%% Initialization
for v = 1:V
    C{v} = zeros(N,N);
    Z{v} = zeros(N, N);
    L{v} = zeros(N, N);
    R{v} = zeros(N, N);
    E{v} = zeros(size(X{v}));
    Q1{v} = zeros(size(X{v}));
    Q2{v} = zeros(N, N);
    Q3{v} = zeros(N, N);
end
%% interation
for iter = 1:maxIter
    %% Update Zv
    for v = 1:V
        M1{v} = X{v} - E{v} + Q1{v}/miu;
        M2{v} = L{v}*C{v}*R{v}' - Q2{v}/miu;
        tempZ{v} = X{v}'*X{v} + eye(N);
        Z{v} = tempZ{v}\(X{v}'*M1{v}+M2{v});
    end

    %% Update Y_tensor
    C_tensor = cat(3, C{:,:});
    c = C_tensor(:);
    Q3_tensor = cat(3, Q3{:,:});
    q3 = Q3_tensor(:);

    [y, ~] = wshrinkObj_tanh(c+q3/miu, 1/miu, sX, 0, 3, alfa, theta);
    Y_tensor = reshape(y, sX);
    for v = 1:V
        Y{v} = Y_tensor(:,:,v);
    end

    %% Update Cv
    for v = 1:V
        N1{v} = Z{v} + Q2{v}/miu;
        N2{v} = Y{v} - Q3{v}/miu;
        C{v} = (L{v}'*N1{v}*R{v}+N2{v})/(lambda2*(LL{v}+LL{v}')/miu+R{v}'*R{v}+eye(N));
    end

    %% Update Lv and Rv
    for v = 1:V
        tempL{v} = (Z{v}+Q2{v}/miu)*R{v}*C{v}';
        [u,~,vv] = svd(tempL{v},'econ');
        L{v} = u*vv';

        tempR{v} = (Z{v}+Q2{v}/miu)'*L{v}*C{v};
        [u,~,vv] = svd(tempR{v},'econ');
        R{v} = u*vv';
    end

    %% Updata Ev
    for v = 1:V
        tempE{v} = X{v} - X{v}*Z{v} + Q1{v}/miu;
        for i = 1:N
            nw = norm(tempE{v}(:,i));
            if nw > lambda1/miu
                x = (nw-lambda1/miu)*tempE{v}(:,i)/nw;
            else
                x = zeros(length(tempE{v}(:,i)),1);
            end
            tempE{v}(:,i) = x;
        end
        E{v} = tempE{v};
    end

    %% Update  Q1, Q2, Q3 and miu
    for v = 1:V
        tempQ1{v} = X{v} - X{v}*Z{v} - E{v};
        tempQ2{v} = Z{v} - L{v}*C{v}*R{v}';
        tempQ3{v} = C{v} - Y{v};
        Q1{v} = Q1{v} + miu*tempQ1{v};
        Q2{v} = Q2{v} + miu*tempQ2{v};
        Q3{v} = Q3{v} + miu*tempQ3{v};
    end
    miu = min(rho*miu,miu_max);

    %% Stop
    for v=1:V
        tempstop = Z{v} - L{v}*C{v}*R{v}';
        temp_ter1(v) = max(max(abs(tempstop)));
    end
    stop = max(temp_ter1);
    error(iter) = stop;
    %------------------------------------------------------------------------
    disp(['iter' num2str(iter) 'stop' num2str(stop)])
    if abs(stop)<0.01
        break
    end

end

%%
Zn = zeros(N, N);
for v = 1 : V
    Zn = Zn + Z{v};
end

end