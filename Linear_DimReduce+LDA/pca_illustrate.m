% Generate a bivariate normal random data set
rng(3); mu = [0 0]; sigma = [1 1.2; 1.2 2];
X = mvnrnd(mu,sigma,200);
close all; clc; figure(1);              % Create a new figure
set(gcf,'Position',[50 50 900 750]);    % Set figure position
set(gcf,'color','w');                   % Set background color to white

% Plot the data as a 2D scatter
subplot(224);
scatter(X(:,1),X(:,2),10,'b','filled'); % Size 10, Color 'blue'
lim = 6; axis(lim*[-1 1 -1 1]);         % Set axis limits
grid on; box on; hold on;
plot([0 0],lim*[-1 1],'k');             % Plot x-axis      
plot(lim*[-1 1],[0 0],'k')              % Plot y-axis

% Plot the distribution of the x1 dimension
subplot(222);
[f,xi] = ksdensity(X(:,1));         % Use Kernel density estimation (KDE)
fill(xi,f,'b','FaceAlpha',0.2);     % Plot the distribution using fill
hold on; plot(xi,f,'b','LineWidth',1.5); % Highlight the distribution
axis([-lim lim 0 1]); axis off; box off; % Remove the axes

% Plot the distribution of the x2 dimension
subplot(223);
[f,xi] = ksdensity(X(:,2));         % Use Kernel density estimation
fill(xi,f,'b','FaceAlpha',0.2);     % Plot the distribution using fill
hold on; plot(xi,f,'b','LineWidth',1.5); % Highlight the distribution
axis([-lim lim 0 1]); axis off; box off; % Remove the axes
camroll(90);                             % Rotate the axes

% Get the first principal component coefficients
% where V = coefficients matrix.
V = pca(X,'Centered',false);
subplot(224);
L = [2*lim 0; -2*lim 0];        % Define some line at 2 arbitrary pts.
L1 = (V*L')'; hold on;          % Project / rotate the 2 pts. using V
plot(L1(:,1),L1(:,2),'r--',...
    'LineWidth',1.5);           % Plot red dashed line (max variance)
col = [34,139,34]/255;          % Forest green color (for dist. later)

% Start iterating over different projection angles
for j = [zeros(1,10), 0:0.05:2*pi]

    % Create the rotation matrix, given angle theta
    coeff = @(theta) [cos(theta), -sin(theta);
                        sin(theta), cos(theta)];

    V = coeff(j);    % Coefficients matrix
    PC = (V*L')';    % Normal to the projection (to plot the dashed line)
                     % PC = principal component

    subplot(224); hold on;
    a = plot(PC(:,1),PC(:,2),'k--',...
             'LineWidth',1.5);              % Plot the rotated dashed line
    scores = (V'*X')';                      % Project actual data                                 
    [f,xi] = ksdensity(scores(:,1));        % Use KDE on actual data
    dist = [xi' 11*f']; dist = (V*dist')';  % Rotate the distribution

    % Plot the green distribution, rotated by theta
    b = fill(dist(:,1),dist(:,2),col,'FaceAlpha',0.2);
    c = plot(dist(:,1),dist(:,2),'Color',col,'LineWidth',1.5);

    % Display the variance of projected actual data
    title(sprintf('Var = %.4f',var(scores(:,1))))
    axis(lim*[-1 1 -1 1]); grid on; box on;

    % Code for creating a GIF
    % exportgraphics(gcf,'PCA_illustrate.gif','Append',true);

    % Reset the figure by deleting the plotted distribution
    pause(0.1); delete(a); delete(b); delete(c);
end
