%% Bayesian Curve Fitting
% Author: Karl Ezra Pilario, 31 Aug 2018
% Reference: Bishop, Pattern Recognition and Machine Learning, 2006

%% Generate training data and set parameters
clear;
N = 30;                                 % Number of data points
x = linspace(0,1,N);                    % Training data, x
t = sin(2*pi*x) + 0.2*randn(size(x));   % Training data, t
X = linspace(-0.5,1.5,100);             % Define finely separated x values
T = sin(2*pi*X);                        % Compute a smooth curve for X

% Set the constant values. Note: precision = 1/variance
a = 5e-3;       % alpha, precision of the weights, w
b = 11.1;       % beta,  precision of the inputs, x
D = 9;          % degree of polynomial to be fitted
M = D+1;        % no. of parameters in the model
W = 0.5*[1 -1]; % no. of std devs for predictive distribution

% Randomize incoming training data points
ind = randperm(N);
t = t(ind); x = x(ind);

z = input('Make a GIF? (y/n): ','s');
if z == 'y', filename = 'bayesiancurvefit.gif'; end
clc;

for j = 1:N
%% Perform sequential curve fitting

    t2 = t(1:j); x2 = x(1:j);               % Collect points 1 to j
    phi = bsxfun(@power,x2',0:D)';          % Design matrix, Eq. (3.16)
    Sinv = a*eye(M) + b*(phi*phi');         % Cov(weights), Eq. (1.72)
    PHI = bsxfun(@power,X',0:D)';           % Finely separated x
    s = zeros(size(X)); m = s;
    for k = 1:length(X)
        m(k) = b*PHI(:,k)'*(Sinv\(phi*t2'));    % Mean, Eq. (1.70)
        s(k) = 1/b + PHI(:,k)'*(Sinv\PHI(:,k));	% Var, Eq. (1.71)
    end
    
%% Plot the results

    f = figure(1); 
    clf; scatter(x2,t2,'b','filled');           % Plot data points 1 to j 
    hold on; grid on; box on;
    axis([-0.5 1.5 -1.5 1.5]);                  % Fix the axes
    plot(X,T,'g',X,m,'r-','LineWidth',2);       % Plot m(x)
    p1 = W(1)*sqrt(s)+m;                        % Solve for +W*s(x)
    p2 = W(2)*sqrt(s)+m;                        % Solve for -W*s(x)
    p = [X fliplr(X); p1 fliplr(p2)];           % Boundary pts for fill
    h = fill(p(1,:),p(2,:),'m');                % Plot +/-W*s(x)
    set(h,'facealpha',0.2);                     % Make fill translucent
    title(sprintf('%d out of %d data points',j,N));
    legend({'Observed Data','Ground Truth',...
        'Mean Estimate','Distribution Estimate'},...
        'Location','southoutside');
    %disp('Press any key to continue...'); 
    pause; clc;
    
    if z == 'y'                                 % For making a GIF
        frame = getframe(f); 
        im = frame2im(frame); 
        [imind,cm] = rgb2ind(im,256);
        if j == 1 
            imwrite(imind,cm,filename,'gif','Loopcount',inf); 
        else 
            imwrite(imind,cm,filename,'gif','WriteMode','append'); 
        end
    end
end

