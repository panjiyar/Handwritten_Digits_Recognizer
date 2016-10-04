function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y);% 118 : number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


[tempJ, tempGrad] = costFunction(theta, X, y);

size(theta);
a = theta.*theta; % 28 by 1
c = theta(1)*theta(1);
b = sum(a);
b= b - c;
b = b*lambda;
b=b/2;
J=b/m;
J = J + tempJ;


d = lambda/m;
e = d*theta;
e(1) = 0;

grad = tempGrad + e;


% =============================================================

end
