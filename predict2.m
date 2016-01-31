function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
hidden_layer_size=size(Theta1,1);
% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

X = [ones(m, 1) X];
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


for n=1:m
    %% Hidden layer
    activation1=zeros(hidden_layer_size,1);
    for a=1:hidden_layer_size
        activation1(a)=sigmoid(X(n,:)*Theta1(a,:)');
    end
    activation1=[1; activation1];
    %% Output layer
    activation2=zeros(num_labels,1);
    for a=1:num_labels
        activation2(a)=sigmoid(Theta2(a,:)*activation1);
    end
    %% Prediction
    [~,p(n)]=max(activation2);
    % =========================================================================
end
p=p(:);
end
