col1 = [34,139,34]/255;     % Forest green color (for dist. later)
col2 = [255,165,0]/255;     % Orange color (for dist. later)
clc; close all;

% Generate P(x) ~ N(0,1)
mu1 = 0; sigma1 = 1;
x = -10:0.05:10; P = normpdf(x,mu1,sigma1); P = P/sum(P);
fill(x,P,col1,'FaceAlpha',0.3); hold on;
plot(x,P,'Color',col1,'LineWidth',1.5);
set(gcf,'color','w');                   % Set background color to white
set(gcf,'Position',[50 50 500 250]);    % Set figure position

mu2 = -2:0.1:2; mu2 = [mu2 flip(mu2)]; rng(0);
sigma2 = interp1(-2:0.5:2,rand(1,9)+0.4,mu2);

% Generate various Q(x) ~ N(mu,sigma)
for j = 1:length(mu2)
    Q = normpdf(x,mu2(j),sigma2(j)); Q = Q/sum(Q);
    a = fill(x,Q,col2,'FaceAlpha',0.3);
    b = plot(x,Q,'Color',col2,'LineWidth',1.5);
    legend({'P(x)','','Q(x)',''},'FontSize',12)
    title(sprintf('D_K_L(P||Q) = %.3f',sum(P.*log(P./Q))),'FontSize',15)
    axis([-4 4 0 0.04]);
    
    % Code for creating a GIF
    exportgraphics(gcf,'KL_illustrate.gif','Append',true);

    pause(0.1); delete(a); delete(b);
end