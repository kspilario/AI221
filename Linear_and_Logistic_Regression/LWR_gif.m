clc; close all; rng(0);

% Generate Sine Data Set
N = 100;                        % No. of training points
x = linspace(0,12,N)';          % Training data, x
y = sin(x) + 0.5*rand(N,1);     % Training data, y
y = (y - mean(y))./std(y);      % Standard scaling on y

scatter(x,y,'b','filled','MarkerFaceAlpha',0.3); 
grid on; box on; set(gcf,'Color','w'); 

xp = linspace(-2,14,200)';      % Finely spaces x values 
yp = zeros(size(xp));

for tau = 0.1:0.02:3
    weighting = @(x,X,tau) exp(-(x-X).^2/(2*tau^2));
    for j = 1:length(xp)
        W = diag(weighting(xp(j),x,tau));
        X = [ones(N,1), x];     
        w = (X'*W*X)\(X'*W*y);  % Locally weighted linear reg.
        yp(j) = [1 xp(j)]*w;    % Predict on xp(j)
    end

    hold on; p = plot(xp,yp,'k','LineWidth',1.5);
    set(gcf,"Position",[488,506.6,560,255.4]);
    title(sprintf('LWR, tau = %.4f',tau))
    axis([-2 14 -3 3]);

    % For creating a GIF
    % exportgraphics(gcf,'LWR_tau.gif','Append',true);

    pause(0.5); delete(p);
end
