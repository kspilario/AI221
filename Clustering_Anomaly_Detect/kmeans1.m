% Sample data set
rng(20);
x = [mvnrnd([1 3],0.5*eye(2),100);
     mvnrnd([-2 1],0.3*eye(2),100);
     mvnrnd([2 -1],0.7*eye(2),100);
     mvnrnd([0 -3],0.4*eye(2),100);
     mvnrnd([-1 5],0.2*eye(2),100)];
   
%% Initialize K-means
x = (x - mean(x))./std(x);              % Normalize the data (optional)
tol = 1e-3;                             % Set a tolerance
K = 5;                                  % Assumed no. of clusters
mu = x(randi(length(x),K,1),:);         % Initial guess of centroids
close all; plotdata(x,mu);              % Plot initial iteration
set(gcf,'color','w');                   % Set background color to white

% Code for creating a GIF
% exportgraphics(gcf,'kmeans1_rng20.gif','Append',true);
 
%% K-means Clustering
mu0 = zeros(size(mu)); iter = 0;        % Save the old centroids here
 
while sum(abs(mu0-mu),'all') > tol
    iter = iter + 1; mu0 = mu;          % Save the old centroids to mu0
    
    % Assign each sample to a centroid
    [~,ind] = min(pdist2(x,mu,'Euclidean'),[],2);
    
    % Compute the new centroids
    mu = cell2mat(arrayfun(@(j) mean(x(ind == j,:)),(1:K)','Uni',false));
    fprintf('Iteration %d:\n',iter);
    disp(mu); plotdata(x,mu);

    % Code for creating a GIF
    % exportgraphics(gcf,'kmeans1_rng20.gif','Append',true);
    pause(0.1);
end
 
%% For Plotting the Data and Voronoi Diagram
function plotdata(x,mu)
    scatter(x(:,1),x(:,2),12,'b','filled');         % Scatter plot of
    box on; axis([-1 1 -1 1]*3); hold on;           %   normalized data
    [vx,vy] = voronoi(mu(:,1),mu(:,2));             % Voronoi edges
    scatter(mu(:,1),mu(:,2),'r','filled');          % Plot the centroids
    plot(vx,vy,'k-','LineWidth',1.2); hold off;     % Plot the edges
end
