function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
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
    %       of the cost function (computeCost) and gradient here.
    %



    %iter
    %theta = theta .- sum((X*theta - y).*X(:,2)) *alpha /m 
    t1 = theta(1) .- (X*theta - y)' * X(:,1) * alpha / m;
    t2 = theta(2) .- (X*theta - y)' * X(:,2) * alpha / m;
    theta = [t1; t2];

    

    %fprintf('Program paused. Press enter to continue.\n');
    %pause;



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    %J_history(iter)

end

end
