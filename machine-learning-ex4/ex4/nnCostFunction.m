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
J = 0;

% Set intermediate totals
J_1 = 0;
J_2 = 0;

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% before we start, need to recode vector y into a matrix Y
Y = zeros(size(y),num_labels);
for i=1:m
  % set the relevant element in the row to 1
  Y(i,y(i))=1;
endfor

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% first calculate hypothesis
% 1. Add the bias a0(1), or x0, to create a1
a1 = cat(2,ones(m,1),X);
% 2. Calculate z2
z2 = a1 * Theta1';
% 3. Calculate the sigmoid and add a bias to get a2
a2 = cat(2,ones(m,1),sigmoid(z2));
% 4. Calculate z3
z3 = a2 * Theta2';
% 5. Calculate the hypothesis
h = sigmoid(z3);

% then calculate the cost of that hypothesis
% Calculate the non-regularised portion
for i=1:m
  for k=1:num_labels
    J_1 += (-1 * Y(i,k) * log(h(i,k))) - ( (1-Y(i,k)) * log(1-h(i,k)) );
  endfor
endfor
J_1 /= m;

% Calculate the regularised term
J_2 = (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

% caclculate J
J = J_1 + J_2;

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

for t = 1:m
  % 1. Calculate a3
  a3(t,:) = h(t,:);
  
  % 2. Calculae d3
  d3 = a3(t,:) - Y(t,:);
  
  % 3. Calculate d2 
  d2 = (d3 * Theta2(:,2:end)) .* sigmoidGradient(z2(t,:));

  % accumulate gradients
  Theta1_grad += d2' * a1(t,:);
  Theta2_grad += d3' * a2(t,:);

endfor

Theta1_grad /= m;
Theta2_grad /= m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------

% =========================================================================

% Add the regularisation terms
Theta1_grad(:, 2:end) += ((lambda / m) .* Theta1(:, 2:end));
Theta2_grad(:, 2:end) += ((lambda / m) .* Theta2(:, 2:end));

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
