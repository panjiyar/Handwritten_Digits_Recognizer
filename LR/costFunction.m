function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % 100 =number of training examples

%y is 100 by 1

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

theta;  %3 by 1  
X;      %100 by 3


a = theta' * X';
h = sigmoid(a); %1 by 100
l = log(h);
p = log(1-h);
size(l);
size(p);
size(h);
size(y);
y;
1-y;
b = (y'.*l);
c = (1-y)'.*p;
d = b+c;

J = sum(d);
J = -J;
J = J/m;



first = h' - y; % 100 by 1
full = first' * X;

full = full';



grad = full/m;








% =============================================================

end
