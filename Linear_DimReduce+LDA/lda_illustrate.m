% Generate two bivariate normal random data sets
rng(2); 
mu = [-3 3]; sigma = [2 1; 1 3];
X1 = mvnrnd(mu,sigma,200);
mu = [3 -2]; sigma = [2 1; 1 3];
X2 = mvnrnd(mu,sigma,200);
close all; clc; figure(1);              % Create a new figure
set(gcf,'Position',[50 50 900 750]);    % Set figure position
set(gcf,'color','w');                   % Set background color to white

% Plot the data sets as a 2D scatter
subplot(224);
scatter(X1(:,1),X1(:,2),10,'b','filled'); % X1, Color 'blue'
hold on; grid on; box on;
scatter(X2(:,1),X2(:,2),10,'m','filled'); % X2, Color 'pink'
lim = 10; axis(lim*[-1 1 -1 1]);          % Set axis limits
plot([0 0],lim*[-1 1],'k');               % Plot x-axis      
plot(lim*[-1 1],[0 0],'k')                % Plot y-axis

% Plot the distribution of the x1 dimension
subplot(222);
[f1,xi1] = ksdensity(X1(:,1));      % Use Kernel density estimation (KDE)
[f2,xi2] = ksdensity(X2(:,1));      % Use KDE
fill(xi1,f1,'b','FaceAlpha',0.2);   % Plot the distribution using fill
hold on; axis off; box off;         % Remove the axes
fill(xi2,f2,'m','FaceAlpha',0.2);   % Plot the distribution using fill
plot(xi1,f1,'b','LineWidth',1.5);   % Highlight the blue distribution
plot(xi2,f2,'m','LineWidth',1.5);   % Highlight the pink distribution
axis([-lim lim 0 1]); 

% Plot the distribution of the x2 dimension
subplot(223);
[f1,xi1] = ksdensity(X1(:,2));      % Use Kernel density estimation (KDE)
[f2,xi2] = ksdensity(X2(:,2));      % Use KDE
fill(xi1,f1,'b','FaceAlpha',0.2);   % Plot the distribution using fill
hold on; axis off; box off;         % Remove the axes
fill(xi2,f2,'m','FaceAlpha',0.2);   % Plot the distribution using fill
plot(xi1,f1,'b','LineWidth',1.5);   % Highlight the blue distribution
plot(xi2,f2,'m','LineWidth',1.5);   % Highlight the pink distribution
axis([-lim lim 0 1]); camroll(90);  % Rotate the axes

% Perform binary LDA to get red dashed line
x1m = mean(X1); x2m = mean(X2); xm = mean([X1; X2]); % Get the means
n1 = length(X1); n2 = length(X2);                    % Get the lengths
Sw = (X1-x1m)'*(X1-x1m) + (X2-x2m)'*(X2-x2m);        % Betw'n-class scatter
Sb = n1*(x1m-xm)'*(x1m-xm) + n2*(x2m-xm)'*(x2m-xm);  % Within-class scatter
[V,D] = eig(Sb,Sw);                                  % Solve Sb*V=D*Sw*V
[~,id] = sort(diag(D),'descend');                    % Sort eigs descending
V = V(:,id); subplot(224);
L = [40*lim 0; -40*lim 0];      % Define some line at 2 arbitrary pts.
L1 = (V*L')'; hold on;          % Project / rotate the 2 pts. using V
plot(L1(:,1),L1(:,2),'r--', ...
    'LineWidth',1.5);           % Plot red dashed line (max variance)
col1 = [34,139,34]/255;         % Forest green color (for dist. later)
col2 = [255,165,0]/255;         % Orange color (for dist. later)

% Start iterating over different projection angles
for j = [zeros(1,10), 0:0.05:2*pi]

    % Create the rotation matrix, given angle theta
    coeff = @(theta) [cos(theta), -sin(theta);
                        sin(theta), cos(theta)];

    V = coeff(j);    % Coefficients matrix
    PC = (V*L')';    % Normal to the projection (to plot the dashed line)
                     % PC = principal component

    subplot(224); hold on;
    a = plot(PC(:,1),PC(:,2),'k--', ...
             'LineWidth',1.5);              % Plot the rotated dashed line

    % For green distribution
    scores1 = (V'*X1')';                    % Project actual data                                 
    [f,xi] = ksdensity(scores1(:,1));       % Use KDE on actual data
    dist = [xi' 20*f']; dist = (V*dist')';  % Rotate the distribution
    b = fill(dist(:,1),dist(:,2),col1,'FaceAlpha',0.2);
    c = plot(dist(:,1),dist(:,2),'Color',col1,'LineWidth',1.5);

    % For orange distribution
    scores2 = (V'*X2')';                    % Project actual data                                 
    [f,xi] = ksdensity(scores2(:,1));       % Use KDE on actual data
    dist = [xi' 20*f']; dist = (V*dist')';  % Rotate the distribution
    d = fill(dist(:,1),dist(:,2),col2,'FaceAlpha',0.2);
    e = plot(dist(:,1),dist(:,2),'Color',col2,'LineWidth',1.5);
    
    % Plot the segment between 2 class means
    M = [mean(scores1(:,1)), 0; mean(scores2(:,1)), 0];
    PC = (V*M')'; f = plot(PC(:,1),PC(:,2),'r-o','LineWidth',2);

    % Display the within-class var. and distance between the 2 means
    title(sprintf('Distance=%.3f\nVar1=%.3f, Var2=%.3f',...
        abs(diff(M(:,1))), var(scores1(:,1)),var(scores2(:,1))));

    % Code for creating a GIF
    % exportgraphics(gcf,'LDA_illustrate.gif','Append',true);

    % Reset the figure by deleting the plotted distribution
    pause(0.1); delete(a); delete(b); delete(c); 
    delete(d); delete(e); delete(f);
end