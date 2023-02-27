clc; clear; close all;
x = rand(200,1);
c = x < 0.5; a = x(c); b = x(~c); subplot(131);
scatter(a,ones(size(a)),15,'b','filled','MarkerFaceAlpha',0.7); hold on;
scatter(b,ones(size(b)),15,'m','filled','MarkerFaceAlpha',0.3); box on;
title(sprintf('Fraction of data\nthat is blue: %.2f%%',...
    length(a)/length(x)*100));
xlabel('Feature');

x = rand(200,2);
c = x(:,1) < 0.5 & x(:,2) < 0.5;
a = x(c,:); b = x(~c,:); subplot(132);
scatter(a(:,1),a(:,2),15,'b','filled','MarkerFaceAlpha',0.7); hold on;
scatter(b(:,1),b(:,2),15,'m','filled','MarkerFaceAlpha',0.3); box on;
title(sprintf('Fraction of data\nthat is blue: %.2f%%',...
    length(a)/length(x)*100));
fill([0 0 0.5 0.5],[0 0.5 0.5 0],'y','FaceAlpha',0.1);
xlabel('Feature 1'); ylabel('Feature 2');

x = rand(200,3);
c = x(:,1) < 0.5 & x(:,2) < 0.5 & x(:,3) < 0.5;
a = x(c,:); b = x(~c,:); subplot(133);
scatter3(a(:,1),a(:,2),a(:,3),15,'b','filled','MarkerFaceAlpha',0.7); hold on;
scatter3(b(:,1),b(:,2),b(:,3),15,'m','filled','MarkerFaceAlpha',0.3); box on;
title(sprintf('Fraction of data\nthat is blue: %.2f%%',...
    length(a)/length(x)*100));
fill3([0 0.5 0.5 0],[0 0 0.5 0.5],[0 0 0 0],'y','FaceAlpha',0.1);
fill3([0 0.5 0.5 0],[0.5 0.5 0.5 0.5],[0 0 0.5 0.5],'y','FaceAlpha',0.1);
fill3([0.5 0.5 0.5 0.5],[0 0.5 0.5 0],[0 0 0.5 0.5],'y','FaceAlpha',0.1);
xlabel('Feature 1'); ylabel('Feature 2'); zlabel('Feature 3');

set(gcf,'Position',[50 50 1200 350]);   % Set figure position
set(gcf,'color','w');                   % Set background color to white
