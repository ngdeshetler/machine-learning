function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
J2=0;
grad = zeros(size(theta));
%Not calculating theta0 seperate
%lambda_list=[0 repmat(lambda, [length(theta)-1,1])];

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i=1:m
    J=J+(-y(i)*log(sigmoid(X(i,:)*theta))-(1-y(i))*log(1-sigmoid(X(i,:)*theta)));
end
J=J/m;

for j=2:length(theta)
    J2=J2+(theta(j)^2);
end
J=J+(J2*(lambda/(2*m)));

summ=0;
for n=1:m
    summ=summ + ((sigmoid(X(n,:)*theta)-y(n))*X(n,1));
end
grad(1)=summ/m;

for i=2:length(theta)
    summ=0;
    for n=1:m
        summ=summ + ((sigmoid(X(n,:)*theta)-y(n))*X(n,i));
    end
    grad(i)=(summ/m)+((lambda/m)*theta(i));
end

%Not calculating theta0 seperate
% for i=1:length(theta)
%     summ=0;
%     for n=1:m
%         summ=summ + ((sigmoid(X(n,:)*theta)-y(n))*X(n,i));
%     end
%     grad(i)=(summ/m)+((lambda_list(i)/m)*theta(i));
% end



% =============================================================

end
