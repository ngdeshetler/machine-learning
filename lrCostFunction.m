function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
%Not calculating theta0 seperate
lambda_list=[0; repmat(lambda, [length(theta)-1,1])];

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


% Unregularized
% for i=1:m
%     J=J+(-y(i)*log(sigmoid(X(i,:)*theta))-(1-y(i))*log(1-sigmoid(X(i,:)*theta)));
% end
% J=J/m;
%J=((J+sum(-y.*log(sigmoid(X*theta))-(1-y).*log(1-sigmoid(X*theta))))/m);

% Regularized
% for j=2:length(theta)
%     J2=J2+(theta(j)^2);
% end
% J=J+(J2*(lambda/(2*m)));
J=((J+sum(-y.*log(sigmoid(X*theta))-(1-y).*log(1-sigmoid(X*theta))))/m)+(sum(theta(2:end).^2)*(lambda/(2*m)));


% summ=0;
% for n=1:m
%     summ=summ + ((sigmoid(X(n,:)*theta)-y(n))*X(n,1));
% end
% grad(1)=summ/m;
% 
% for i=2:length(theta)
%     summ=0;
%     for n=1:m
%         summ=summ + ((sigmoid(X(n,:)*theta)-y(n))*X(n,i));
%     end
%     grad(i)=(summ/m)+((lambda/m)*theta(i));
% end

%Not calculating theta0 seperate
% for i=1:length(theta)
%     summ=0;
%     for n=1:m
%         summ=summ + ((sigmoid(X(n,:)*theta)-y(n))*X(n,i));
%     end
%     
%     grad(i)=(summ/m)+((lambda_list(i)/m)*theta(i));
% end

grad=((lambda_list/m).*theta)+((X'*(sigmoid(X*theta)-y))/m);




% =============================================================

grad = grad(:);

end
