
% See Section 3.2 in user's manual for details.

clear; clc;

% The data
[X1,X2] = meshgrid(-1:0.2:1, -1:0.2:1);
X(:,1) = reshape(X1, numel(X1), 1);
X(:,2) = reshape(X2, numel(X2), 1);
clear X1 X2;
Y = sin(0.83*pi*X(:,1)) .* cos(1.25*pi*X(:,2));
Xt = rand(10000,2);
Yt = sin(0.83*pi*Xt(:,1)) .* cos(1.25*pi*Xt(:,2));

%%

% Parameters
params = aresparams2('maxFuncs', 101, 'c', 0, 'maxInteractions', 2); % piecewise-cubic
%params = aresparams2('maxFuncs', 101, 'c', 0, 'maxInteractions', 2, 'cubic', false); % piecewise-linear

% Building the model
disp('Building the model ==================================================');
[model, ~, resultsEval] = aresbuild(X, Y, params);
model

% Plotting model selection from the backward pruning phase
figure;
hold on; grid on; box on;
h(1) = plot(resultsEval.MSE, 'Color', [0 0.447 0.741]);
h(2) = plot(resultsEval.GCV, 'Color', [0.741 0 0.447]);
numBF = numel(model.coefs);
h(3) = plot([numBF numBF], get(gca, 'ylim'), '--k');
xlabel('Number of basis functions');
ylabel('MSE, GCV');
legend(h, 'MSE', 'GCV', 'Selected model');

% Plotting the model
aresplot(model);

% Info on the basis functions
disp('Info on the basis functions =========================================');
aresinfo(model, X, Y);

% Printing the model
disp('The model ===========================================================');
areseq(model, 5);

% Testing on test data
disp('Testing on test data ================================================');
results = arestest(model, Xt, Yt)
