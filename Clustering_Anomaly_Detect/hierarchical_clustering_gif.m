%% Sample Data Set
x = [0.225, 0.821; 0.329, 0.785; 0.253, 0.706; 0.149, 0.693;
     0.359, 0.610; 0.412, 0.883; 0.756, 0.696; 0.706, 0.788;
     0.622, 0.842; 0.622, 0.749; 0.556, 0.686; 0.687, 0.584;
     0.557, 0.255; 0.477, 0.364; 0.412, 0.208; 0.555, 0.145;
     0.666, 0.336; 0.356, 0.285; 0.348, 0.431; 0.296, 0.187;
     0.810, 0.480; 0.532, 0.509; 0.262, 0.602; 0.245, 0.297;
     0.481, 0.186; 0.424, 0.095; 0.382, 0.707; 0.659, 0.672;
     0.540, 0.801; 0.767, 0.838];
 
%% Create the linkage data
%  Choices: 'single', 'complete', 'average', 'centroid'
Z = linkage(x,'average');
set(gcf,'Position',[132.2,361.8,1045.6,405.2]);
set(gcf,'color','w'); 
 
%% Perform Hierarchical Clustering
for nClus = length(x):-1:2 
    
    % Plot of the dendogram colored accrdg to no. of clusters (nClus)
    subplot(122);
    color = Z(end-nClus+2,3)-eps;
    T = cluster(Z,'MaxClust',nClus);
    h = dendrogram(Z,0,'Orientation','left','ColorThreshold',color); 
    axis square; box on; set(h,'LineWidth',1.2);
    
    % Plot the data points and the Delaunay Triangulation of clusters
    subplot(121);
    scatter(x(:,1),x(:,2),25,'b','filled'); hold on;
    for j = 1:length(x)
        text(x(j,1)-0.01,x(j,2)+0.02,num2str(j));
    end
    for j = 1:max(T)
        if sum(T == j) == 2
           plot(x(T == j,1),x(T == j,2),'k','Linewidth',1.2);
        elseif sum(T == j) > 2
           tri = delaunay(x(T == j,1),x(T == j,2));
           triplot(tri,x(T == j,1),x(T == j,2),'k','Linewidth',1.2);
        end
    end
    axis([0 1 0 1]); axis square;
    box on; hold off;
    
    % Code for creating a GIF
    % exportgraphics(gcf,'average_link.gif','Append',true);
    pause(0.1);
end
