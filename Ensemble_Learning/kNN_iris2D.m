% Fisher Iris Data Set then PCA
[x,y] = iris_dataset;                   % Load iris data
[~,x] = pca(x');                        % Perform PCA on x
[y,~] = find(y == 1);                   % Reverse one-hot encoding
x = x(:,1:2);                           % Retain first 2 PCs
x = (x - mean(x))./std(x);              % Normalize x to be sure
K = [1, 5, 10];                         % No. of nearest neighbors

close all; figure(1); col = 'bmg'; 
t = -3:0.1:3; [X,Y] = meshgrid(t);
Z = zeros(size(X)); Z1 = Z;              % For decision boundary
for i = 1:length(K)     
    % Perform kNN for each point in Z
    for j = 1:length(X)
        for k = 1:length(X)
            d = [pdist2([t(k) t(j)],x)' y]; % Distance to all x
            d = sortrows(d,1);              % Sort to get nearest
            Z(j,k) = mode(d(1:K(i),2));     % Classify Z(j,k)
        end
    end
    Xv = X(:); Yv = Y(:); Zv = Z(:);
    subplot(1,length(K),i);
    if K(i) == 1, Z1 = Zv; end
     
    for j = 1:3
        scatter(Xv(Zv == j),Yv(Zv == j),15,col(j),...
            'filled','MarkerFaceAlpha',0.2); hold on;
        scatter(x(y == j,1),x(y == j,2),20,col(j),...
            'filled','MarkerEdgeColor','k');
    end
    box on; axis([-3 3 -3 3]);
end
set(gcf,'Color','w');
set(gcf,'Position',[114.6,452.2,1272.8,309.8]);

figure(2);
for j = 1:3
    scatter(Xv(Z1 == j),Yv(Z1 == j),15,col(j),...
        'filled','MarkerFaceAlpha',0.2); hold on;
    scatter(x(y == j,1),x(y == j,2),20,col(j),...
        'filled','MarkerEdgeColor','k'); hold on;
end
[vx,vy] = voronoi(x(:,1),x(:,2));             % Get Voronoi edges
plot(vx,vy,'k-','LineWidth',0.3); hold off;   % Plot the edges
box on; axis([-3 3 -3 3]);
set(gcf,'Color','w');