% Generate spiral data set
rng(42); N = 200; X = zeros(2*N,2); % 200 samples per class
y = repmat([0, 1],N,1); y = y(:);   % True class labels 
radius0 = linspace(0,2,N);          % radius and angle for blue spiral 
theta0 = linspace(0,3*pi,N) + ...
    linspace(0,0.7,N).*randn(1,N);
radius1 = linspace(0,2,N);          % radius and angle for pink spiral 
theta1 = linspace(pi,4*pi,N) + ...
    linspace(0,0.7,N).*randn(1,N);
r = [radius0, radius1]'; t = [theta0, theta1]';
X(:,1) = r.*cos(t); X(:,2) = r.*sin(t);   % x1,x2-coordinate 

clc; close all;

% Iterate on various kernel widths from 5 to 0.2
for kw = sort(linspace(0.2,5,50),'descend')
    mdl = fitcsvm(X,y,'KernelScale',kw,'BoxConstraint',5,...
        'KernelFunction','rbf');                        % Fit an SVM
    [Xf,Yf] = meshgrid(linspace(-3,3,100));             % Make grid of pts
    Xf = Xf(:); Yf = Yf(:);
    Z = predict(mdl,[Xf Yf]);                           % Predict on grid
    gscatter(Xf,Yf,Z,'bm'); hold on;                    % Plot predictions
    scatter(X(y == 0,1),X(y == 0,2),25,'b','filled');   % Spiral (blue pts)
    scatter(X(y == 1,1),X(y == 1,2),25,'m','filled');   % Spiral (pink pts)
    box on; axis([-3 3 -3 3]); set(gcf,'Color','w');
    title(sprintf('Binary SVM, Kernel Width = %.3f',kw));
    xlabel(''); ylabel(''); axis("image"); legend off; 

    % For creating a GIF
    % exportgraphics(gcf,'SVM_kw_illustrate.gif','Append',true);
    pause(0.1);
end