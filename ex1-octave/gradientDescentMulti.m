function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    t1 = theta(1) .- (X*theta - y)' * X(:,1) * alpha / m;
    t2 = theta(2) .- (X*theta - y)' * X(:,2) * alpha / m;
    t3 = theta(2) .- (X*theta - y)' * X(:,3) * alpha / m;

    theta = [t1; t2; t3];









    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
    % J_history(iter)
    % fprintf('Program paused. Press enter to continue.\n');
    % pause;

end

% J_history
% figure;
% plot(J_history);
% fprintf('Program paused. Press enter to continue.\n');
% pause;

end
