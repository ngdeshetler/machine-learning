function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
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
    
    tempj0=theta(1)-alpha*part_deriv(X, y, theta,1)/m;
    tempj1=theta(2)-alpha*part_deriv(X, y, theta,2)/m;
    theta(1)=tempj0;
    theta(2)=tempj1;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
function dJ=part_deriv(X, y, theta,i)
m = length(y);
dJ = 0;
for n=1:m
    dJ=dJ+((theta(1)*X(n,1)+theta(2)*X(n,2))-y(n))*X(n,i);
end
    
end