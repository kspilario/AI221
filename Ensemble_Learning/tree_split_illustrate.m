close all; clear; clc;

rng(0);

% Two Moons
N = 100; x1 = rand([N 1])*4-1; x2 = rand([N 1])*4-3;
m = [x1, 2-0.5*(x1-1).^2]; m = m + 0.8*(rand(size(m))-1);
b = [x2, 0.5*(x2+1).^2-1]; b = b + 0.8*(rand(size(b))-1);
axl = [-4 3 -2 2];

% Concentric Bivariate Gaussians
% N = 100; b = mvnrnd([0 0],2.5*eye(2),2*N);
% c = vecnorm(b')' < 1.5;
% m = b(c,:); b = b(~c,:);
% axl = [-6 6 -6 6];

% Separable Bivariate Gaussians
% N = 100;
% b = mvnrnd([-2 0],1.5*eye(2),N); 
% m = mvnrnd([2 0],1.5*eye(2),N);
% axl = [-6 6 -6 6];

% Gather all (x,y) points and their target classes, t: [x, y, t]
x = [b(:,1); m(:,1)];
y = [b(:,2); m(:,2)];
t = [zeros(1,length(b)) ones(1,length(m))]+1;

subplot(4,4,[1:2, 5:6])
scatter(b(:,1),b(:,2),'b','filled','MarkerFaceAlpha',0.2); hold on;
scatter(m(:,1),m(:,2),'m','filled','MarkerFaceAlpha',0.2);
axis(axl); grid on; box on; set(gcf,'Color','w');

data = [x y t'];

data = sortrows(data,1);
[~,gini_id] = find_split(data(:,1),data(:,3),axl,1);
subplot(4,4,[1:2, 5:6]); plot(data(gini_id,1)*[1 1],axl(3:4),'k--');

data = sortrows(data,2);
[~,gini_id] = find_split(data(:,2),data(:,3),axl,2);
subplot(4,4,[1:2, 5:6]); plot(axl(1:2),data(gini_id,2)*[1 1],'k--');

function [min_gini,gini_id] = find_split(x,t,axl,feat)

    th = 90; N = length(x);
    merr_1 = zeros(1,N); gini_1 = zeros(1,N); enpy_1 = zeros(1,N);
    merr_2 = zeros(1,N); gini_2 = zeros(1,N); enpy_2 = zeros(1,N);
    node1 = [0 0]; node2 = sum(t==[1 2]); % Counter for [b m] in each node
    gini = zeros(1,2*N); ctr = 1; 
    
    for split = x'
        w = ctr/N; subplot(4,4,[1:2, 5:6])
        if feat == 1
            h = plot([1 1]*split,axl(3:4),'k','LineWidth',0.8);
        else 
            h = plot(axl(1:2),[1 1]*split,'k','LineWidth',0.8);
        end
    
        node1(t(ctr)) = node1(t(ctr)) + 1;
        node2(t(ctr)) = node2(t(ctr)) - 1;
    
        p_1 = node1(1)/sum(node1); % Prop. of b in node1 (left)
        p_2 = node2(1)/sum(node2); % Prop. of b in node2 (right)
    
        merr_1(ctr) = 1-max(p_1,1-p_1);
        merr_2(ctr) = 1-max(p_2,1-p_2);
        gini_1(ctr) = 2*p_1*(1-p_1);
        gini_2(ctr) = 2*p_2*(1-p_2);
        enpy_1(ctr) = -p_1*log2(p_1+eps) - (1-p_1)*log2(1-p_1+eps);
        enpy_2(ctr) = -p_2*log2(p_2+eps) - (1-p_2)*log2(1-p_2+eps);
        gini(ctr) = w*gini_1(ctr) + (1-w)*gini_2(ctr);
    
        if feat == 1, subplot(4,4,9:10);
        else,         subplot(4,4,[3 7]); end
        h1 = plot(x(1:ctr),merr_1(1:ctr),'LineWidth',1.5,'Color',...
            [187 222 251]/255,'DisplayName','Misclassification Error (Left)'); 
        hold on;
        h2 = plot(x(1:ctr),gini_1(1:ctr),'LineWidth',1.5,'Color',...
            [0 128 255]/255,'DisplayName','Gini Index (Left)'); 
        h3 = plot(x(1:ctr),enpy_1(1:ctr),'b','LineWidth',1.5,...
            'DisplayName','Cross-Entropy (Left)');
    
        h4 = plot(x(1:ctr),merr_2(1:ctr),'LineWidth',1.5,'Color',...
            [255 185 243]/255,'DisplayName','Misclassification Error (Right)');
        h5 = plot(x(1:ctr),gini_2(1:ctr),'LineWidth',1.5,'Color',...
            [255 115 232]/255,'DisplayName','Gini Index (Right)'); 
        h6 = plot(x(1:ctr),enpy_2(1:ctr),'m','LineWidth',1.5,...
            'DisplayName','Cross-Entropy (Right)');
        hold off; box on; grid on;
        ylabel('Node Impurity');
        [min_gini,gini_id] = min(nonzeros(gini));
        if feat == 1, axis([axl(1:2) 0 1]);
        else, axis([axl(3:4) 0 1]); camroll(th); end

        if feat == 1, subplot(4,4,13:14);
        else,         subplot(4,4,[4 8]); end

        plot(x(1:ctr),gini(1:ctr),'k','LineWidth',1.5); hold on;
        h7 = plot(x(gini_id)*[1 1],[0 0.5],'k--','DisplayName',...
            'Best Split (Min. weighted Gini)'); hold off;
        grid on; box on; ylabel('Weighted Gini');
        if feat == 1, axis([axl(1:2) 0 0.5]);
        else, axis([axl(3:4) 0 0.5]); camroll(th); end

        hL = legend([h1, h2, h3, h4, h5, h6, h7]); 
        hL.Position = [0.541,0.199,0.321,0.23];

        % Create a GIF
        % exportgraphics(gcf,'tree_split.gif','Append',true);

        pause(0.1); ctr = ctr + 1; delete(h);
    end
end