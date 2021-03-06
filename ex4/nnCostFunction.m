function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Calculate Cost Function
os = ones(m, num_labels);

a2 = sigmoid([ones(m, 1) X] * Theta1');
a3 = sigmoid([ones(m, 1) a2] * Theta2');

y_real_result = zeros(m, num_labels);
for j = 1:num_labels    
    y_real_result(:, j) = y == j;    
end

r_matrix = (-y_real_result .* log(a3) - (os - y_real_result) .* log(os - a3));
sum_J = sum(sum(r_matrix)) / m;

% Regularized cost function
Theta1_short = Theta1;
Theta1_short(:, 1) = [];
Theta2_short = Theta2;
Theta2_short(:, 1) = [];

regu = (sum(sum(Theta1_short .^2)) + sum(sum(Theta2_short .^2))) * lambda / (2 * m);

J = sum_J + regu;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
for i=1:m
    delta3 = a3(i, :) - y_real_result(i, :);
    %     regulation
    delta2_regulation = Theta2 * lambda;
    delta2_regulation(:, 1) = 0;
    %     regulation - end
    Theta2_grad = Theta2_grad + delta3' * [1 a2(i, :)] + delta2_regulation;
    
    a2_vector = a2(i, :)';
    value =  a2_vector .* (1 - a2_vector);
    delta2 = Theta2' * delta3' .* ([1; value]);
    delta2 = delta2(2:end);
    %     regulation
    delta1_regulation = Theta1 * lambda;
    delta1_regulation(:, 1) = 0;
    %     regulation - end
    Theta1_grad = Theta1_grad + delta2 * [1 X(i, :)] + delta1_regulation;
end
Theta2_grad = Theta2_grad / m;
Theta1_grad = Theta1_grad / m;

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
