% This is an comparison between Rosenblatt's Perceptron and SVM
% from Cristianini and Shawe-Taylor (2000). "Support Vector Machines
% and Other Kernel-based Learning Methods".

% See Table 2.1: The Perceptron Algorithm (primal form) with a 
% slight change in the definition of a mistake (see "if" statement).

% See Proposition 6.1 for the maximum-margin classifier (SVM).

clc; close all;
x = [0.248,	0.564; 0.103, 0.637; 0.119, 0.489;
     0.458, 0.281; 0.268, 0.378; 0.267, 0.253;
     0.156, 0.311; 0.077, 0.350; 0.109, 0.191;
     0.344, 0.133; 0.494, 0.110; 0.360, 0.886;
     0.492, 0.740; 0.523, 0.895; 0.719, 0.820;
     0.926, 0.764; 0.714, 0.603; 0.548, 0.566;
     0.746, 0.335; 0.852, 0.547];               % Data set: [x1, x2]
y = ones(20,1); y(1:11) = -1;                   % Data set: y = {+1,-1}

%% Rosenblatt's Perceptron
w = [0; 0]; b = 0;          % Initialize parameters w, b
R = max(vecnorm(x'));       % Distance of farthest data pt. from origin
eta = 0.1; k = Inf;         % eta = learning rate
[X,Y] = meshgrid(0:0.01:1); % For plotting the decision boundary
while k > 0
    k = 0;                  % Reset counter for no. of mistakes
    for j = 1:size(x,1)
        if y(j)*(w'*x(j,:)' + b) < 1    % This is "< 0" in the book
            w = w + eta*y(j)*x(j,:)';   % Update w
            b = b + eta*y(j)*R^2;       % Update b
            k = k + 1;                  % Increment counter
        end
    end
end

Z = reshape([X(:) Y(:)]*w + b,size(X));
contourf(X,Y,Z,50,'FaceAlpha',0.3,'LineColor','none');
colormap(redblue); hold on; set(gcf,'Color','w');
scatter(x(y == 1,1),x(y == 1,2),100,'rx','LineWidth',2.5);
scatter(x(y == -1,1),x(y == -1,2),40,'bo','filled');
box on; grid on;
x2 = [(-b+w(1))/w(2), -(b+w(1))/w(2)];
plot([-1 1],x2,'k-','LineWidth',1.5);       % <w,x> + b = 0
x2 = [(1-b+w(1))/w(2), (1-b-w(1))/w(2)];
plot([-1 1],x2,'k--','LineWidth',1.5);      % <w,x> + b = 1
x2 = [(-1-b+w(1))/w(2), -(1+b+w(1))/w(2)];
plot([-1 1],x2,'k--','LineWidth',1.5);      % <w,x> + b = -1

%% Maximum margin classifier
H = eye(3); H(3,3) = 0; f = [0 0 0]'; 
A = -[y.*x, y]; B = -ones(size(y));
wb = quadprog(H,f,A,B);
x2 = [(-wb(3)+wb(1))/wb(2), -(wb(3)+wb(1))/wb(2)];
plot([-1 1],x2,'m-','LineWidth',1.5);       % <w,x> + b = 0
x2 = [(1-wb(3)+wb(1))/wb(2), (1-wb(3)-wb(1))/wb(2)];
plot([-1 1],x2,'m--','LineWidth',1.5);      % <w,x> + b = 1
x2 = [(-1-wb(3)+wb(1))/wb(2), -(1+wb(3)+wb(1))/wb(2)];
plot([-1 1],x2,'m--','LineWidth',1.5);      % <w,x> + b = -1
title(sprintf(['Perceptron: w = [%.3f; %.3f], b = %.3f, Margin = %.3f',...
              '\nSVM: w = [%.3f; %.3f], b = %.3f, Margin = %.3f'],...
              w, b, 1./norm(w), wb', 1./norm(wb(1:2)))); 
legend({'','Positive samples','Negative samples',...
        'Perceptron','','','SVM','',''},...
        'Location','eastoutside','Box','off');
xticks(0:0.1:1); axis equal; axis([0 1 0 1]);

%% Auxiliary Function
function c = redblue
% redblue gives a colormap within [cl(1) cl(2)] from
%   Blue [0 0 1] to White [0 0 0] to Red [1 0 0]
    m = size(get(gcf,'colormap'),1);        % Get size of colormap
    cl = clim;                              % Get colormap limits
    m1 = round(-m*cl(1)/diff(cl));          % Get ratio of (-1), blue
    m2 = m - m1;                            %  the rest are (+1), red
    up = (0:m1-1)'/max(m1-1,1);             % up = [0,...,1] 
    dn = (m2-1:-1:0)'/max(m2-1,1);          % dn = [1,...,0]
    r = [up; ones(m2,1)];                   % red vector
    g = [up; dn];                           % green vector
    b = [ones(m1,1); dn];                   % blue vector
    c = [r g b];
end
