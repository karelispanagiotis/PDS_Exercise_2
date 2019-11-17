% Change the values below and X, Y to create a test case
k = 1; n = 3; m = 2; d = 2;

X = [0 0; 1 1; 9 9];  % n x d
Y = [0 0; 10 10];     % m x d

D = sqrt(sum(X.^2,2) - 2 * X*Y.' + sum(Y.^2,2).');  %n x m
D = D';                                             %m x n

[ndist, nidx] = sort(D, 2);

ndist = ndist(:, 1:k)  % m x k, the k nearest distances for each one of the m query points
nidx = nidx(:, 1:k)    % m x k, the corresponding indices of the above distances