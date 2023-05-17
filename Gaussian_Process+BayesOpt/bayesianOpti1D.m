%% Bayesian Optimization for 1D opti problem

close all; clc; 
Nstart = 1;     % Initial no. of observations
obs = 19;       % No. of more observations to perform

% Example Function
f = @(x) sin(2*x) + cos(3*x) - exp(0.2*x);
xrange = [0 10]; % Search range for x

disp('Function: '); disp(f);
fprintf('Enter [1] if function is to be maximized\n');
fprintf('      [0] if function is to be minimized\n');
tobemax = input('Choice: ');

x_fine = linspace(xrange(1),xrange(2),300);
x = rand(1,Nstart)*diff(xrange)+xrange(1);
y_true = zeros(size(x));

for t = 1:length(x), y_true(t) = f(x(t)); end

for j = 0:obs
    mdl = fitrgp(x',y_true','Sigma',0.1,'ConstantSigma',true,...
        'KernelFunction','squaredexponential');
    % Kernel function was recommended in https://arxiv.org/pdf/1206.2944.pdf
    
    [y_pred,sd,ci] = predict(mdl,x_fine');
    
    subplot(212);
    plot(x_fine,f(x_fine),'k','LineWidth',1.2); hold on;
    p = [x_fine fliplr(x_fine); ci(:,1)' fliplr(ci(:,2)')];
    h = fill(p(1,:),p(2,:),'m','LineStyle','None');     % Conf. bands
    set(h,'facealpha',0.15); grid on; box on;
    plot(x_fine,y_pred,'r','LineWidth',1);              % GPR prediction
    scatter(x,y_true,20,'r','filled',...
        'MarkerEdgeColor','k');                         % Plot seen data
    title(sprintf('No. of observed points: %d',length(x)));
    hold off;
    
    %% Expected Improvement
    % This EI is from http://krasserm.github.io/2018/03/21/bayesian-optimization/
    
    xi = 2;  % Exploration-exploitation parameter (greek letter, xi)
             % High xi = more exploration
             % Low xi = more exploitation (can be < 0)
    
    if tobemax, d = y_pred - max(y_true) - xi; % (y - f*) if maximization
    else,       d = min(y_true) - y_pred - xi; % (f* - y) if minimiziation
    end

    EI = (sd ~= 0).*(d.*normcdf(d./sd) + sd.*normpdf(d./sd));  
    
    subplot(211);
    plot(x_fine,EI,'b','LineWidth',1.2); hold on;
    posEI = find(EI == max(EI));
    xEI = x_fine(posEI(randi(length(posEI))));
    scatter(xEI,max(EI),'b','filled','MarkerEdgeColor','k');
    title('Expected Improvement'); grid on; hold off;
    x(end+1) = xEI;             %#ok<SAGROW> Save xEI as next
    y_true(end+1) = f(x(end));  %#ok<SAGROW> Sample the obj. at xEI
    
    subplot(212); hold on;
    plot(x(end)*[1 1],[-10 5],'k--','Linewidth',1.2); 
    legend({'Ground Truth','95% confidence','Mean Prediction','Observed Data'},...
        'Location','northeast'); 
    hold off; axis([xrange -10 5]); pause(0.5); 
    
end

if tobemax, [ae,be] = max(y_pred); 
            [ao,bo] = max(y_true); str = 'Maximum';
else,       [ae,be] = min(y_pred);
            [ao,bo] = min(y_true); str = 'Minimum';
end
fprintf('Bayesian Optimization\n');
fprintf('  %s (estimated): y(%.6f) = %.6f\n',str,x_fine(be),ae);
fprintf('  %s (observed) : y(%.6f) = %.6f\n',str,x(bo),ao);
