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
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

y2 = zeros(num_labels,m);


for i = 1:m
  y2(y(i),i) = 1; 
end


size(nn_params);          %10285 by 1
size(input_layer_size);   %just a number   
size(hidden_layer_size);  %just a number  
size(num_labels);         %just a number  
size(X);                  %5000 by 400
size(y);                  %5000 by 1
size(lambda);             %just a number  
size(Theta1);             %25 by 401
size(Theta2);             %10 by 26
X = [ones(m,1) X];
size(y2);                 %10 by 5000

input_layer_size; %400ex4
hidden_layer_size;%25
num_labels;       %10
lambda;           %0
m;                %5000
z2 = Theta1*X';
a2 = sigmoid(z2);

size(a2);         %25 by 5000
a2 = a2';
a2 = [ones(m,1) a2];

z3 = Theta2*a2';
a3 = sigmoid(z3);
size(a3);         %10 by 5000

a3;

h = log(a3);
l = log(1-a3);

ab= h.*y2 + l.*(1-y2);

J = sum(ab);
J = sum(J);
J = -J;
J = J/m;



% -------------------------------------------------------------


a = Theta1.*Theta1;
b = Theta2.*Theta2;
a = sum(a);
b = sum(b);

bias1 = a(1,1);
bias2 = b(1,1);

a = sum(a)+sum(b) - bias1 - bias2;
a = a*lambda;
a = a/2;
a = a/m;

J = J + a;



% =========================================================================


z2 = [ones(1,m); z2];

delta3 = a3 - y2;
delta2 = (Theta2'*delta3).*sigmoidGradient(z2);

size(Theta2_grad);             %10 by 26
size(delta3);   %10 by 5000
size(a2);       %5000 by 26
Theta2_grad = delta3*a2;






t1 = size(delta2);   %26 by 5000
op = t1(1)-1;
new_del = ones(op, t1(2));
for i=1:op
for j=1:t1(2)
new_del(i,j) = delta2(i+1,j);
end
end
size(X);        %5000 by 401


size(Theta1_grad);             %25 by 401



Theta1_grad = new_del*X;



Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;

a = Theta1;
b = Theta2;
a = a*lambda;
b = b*lambda;
a = a/m;
b = b/m;
for i=1:t1(1)-1
a(i,1) = 0;
end
t2 = size(b);
for i=1:t2(1)  
b(i,1) = 0;
end

Theta1_grad = Theta1_grad + a;
Theta2_grad = Theta2_grad + b;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
