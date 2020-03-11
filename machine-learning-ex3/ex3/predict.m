function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

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

% 6. Take the max probability on each row
[value, index] = max( h, [], 2 );

% 7. pass indexes (as the predicted digits) to the output
p = index;

% =========================================================================

end
