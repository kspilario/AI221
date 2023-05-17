%% Bayesian Optimization for 2D opti problem

close all; clc; angle = [45 45]; 
Nstart = 10;    % Initial no. of observations
obs = 40;       % No. of more observations to perform

% sin + cos function
% f = @(x) sin(x(1)*2) + cos(x(2)*2+pi/2);
% xrange = [-3 3;         % Search range for x1
%           -3 3];        % Search range for x2

% Black-box Peaks Function
f = @(x) peaks(x(1),x(2))-peaks(x(1)+1,x(2)+1)+2*peaks(x(1)-1,x(2)-1);
xrange = [-3 3;         % Search range for x1
          -3 3];        % Search range for x2

disp('Function: '); disp(f);
fprintf('Enter [1] if function is to be maximized\n');
fprintf('      [0] if function is to be minimized\n');
tobemax = input('Choice: ');

[Xfine,Yfine] = meshgrid(linspace(xrange(1,1),xrange(1,2)),...
                         linspace(xrange(2,1),xrange(2,2)));
subplot(221); xyfine = [Xfine(:), Yfine(:)]; 
z = arrayfun(@(j) f(xyfine(j,:)),1:length(xyfine));
zlim = [floor(min(z)/5)*5, ceil(max(z)/5)*5];       % z-axis ticks by 5
surf(Xfine,Yfine,reshape(z,size(Xfine))); box on;   % Plot ground truth
title('Ground Truth Function to be Optimized');
shading interp; view(angle); axis([xrange([1 3 2 4]), zlim]);

zmax = max(z);
x = [rand(Nstart,1)*diff(xrange(1,:))+xrange(1,1),...
     rand(Nstart,1)*diff(xrange(2,:))+xrange(2,1)]; % Random x positions
y_true = zeros(size(x,1),1);
for t = 1:length(x), y_true(t) = f(x(t,:)); end     % Sample the func at x

for j = 0:obs
    mdl = fitrgp(x,y_true,'KernelFunction','ardmatern52');
    % ardmatern52 kernel was recommended in https://arxiv.org/pdf/1206.2944.pdf
    
    xyfine = [Xfine(:), Yfine(:)];
    [y_pred,sd] = predict(mdl,xyfine);
    
    subplot(222);
    surf(Xfine,Yfine,reshape(y_pred,size(Xfine))); 	% GPR prediction
    shading interp; hold on; view(angle);
    scatter3(x(:,1),x(:,2),y_true,10,'g','filled',...
        'MarkerEdgeColor','k');                     % Plot seen data
    title(sprintf('No. of observed points: %d',length(x)));
    axis([xrange([1 3 2 4]), zlim]); box on; hold off;
    
    subplot(223);
    surf(Xfine,Yfine,reshape(sd,size(Xfine))); 	% Uncertainty (std. dev.)
    shading interp; hold on; view(angle);
    axis([xrange([1 3 2 4]), 0, max(2,max(sd))]);
    title('Uncertainty'); box on; hold off;

    %% Expected Improvement
    % This EI is from http://krasserm.github.io/2018/03/21/bayesian-optimization/
    
    xi = 2;  % Exploration-exploitation parameter (greek letter, xi)
             % High xi = more exploration
             % Low xi = more exploitation (can be < 0)
    
    if tobemax, d = y_pred - max(y_true) - xi; % (y - f*) if maximization
    else,       d = min(y_true) - y_pred - xi; % (f* - y) if minimiziation
    end

    EI = (sd ~= 0).*(d.*normcdf(d./sd) + sd.*normpdf(d./sd));

    [eimax,posEI] = max(EI); xEI = xyfine(posEI,:);
    x(end+1,:) = xEI;               %#ok<SAGROW> Save xEI as next
    y_true(end+1) = f(x(end,:));    %#ok<SAGROW> Sample the obj. at xEI

    subplot(224);
    surf(Xfine,Yfine,reshape(EI,size(Xfine))); 
    shading interp; hold on;
    plot3(xEI([1 1]),xrange(2,:),eimax*[1 1],'--k','LineWidth',1.5);
    plot3(xrange(1,:),xEI([2 2]),eimax*[1 1],'--k','LineWidth',1.5);
    title('Expected Improvement'); grid on; hold off; 
    view(angle); axis(xrange([1 3 2 4])); box on;
    
    subplot(222); hold on; xlabel('x1'); ylabel('x2');
    scatter3(x(end,1),x(end,2),zmax,'m','filled','MarkerEdgeColor','k'); 
    plot3(xEI([1 1]),xrange(2,:),zmax*[1 1],'--k','LineWidth',1.5);
    plot3(xrange(1,:),xEI([2 2]),zmax*[1 1],'--k','LineWidth',1.5);
    hold off; axis(xrange([1 3 2 4])); view(angle); pause(0.5);
    
end

if tobemax, [ae,be] = max(y_pred); 
            [ao,bo] = max(y_true); str = 'Maximum';
else,       [ae,be] = min(y_pred);
            [ao,bo] = min(y_true); str = 'Minimum';
end
fprintf('Bayesian Optimization\n');
fprintf('  %s (estimated):\n\ty(%.6f,%.6f) = %.6f\n',...
    str,xyfine(be,1),xyfine(be,2),ae);
fprintf('  %s (observed):\n\ty(%.6f,%.6f) = %.6f\n',...
    str,x(bo,1),x(bo,2),ao);