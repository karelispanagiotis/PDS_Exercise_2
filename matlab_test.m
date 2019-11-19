% Change the values below and X, Y to create a test case
n = 5;
m = 2;
d = 2;
k = 3;

X = [1 2 1 5 8; 9 1.2 5 6 3.25];  % d x n
Y = [8 4; 1.54 8.225];     % d x m

D = sqrt(sum(X.^2).' - 2 * X.' * Y + sum(Y.^2));  % n x m

[ndist, nidx] = sort(D);

ndist = ndist(1:k,:)  % m x k, the k nearest distances for each one of the m query points
nidx = nidx(1:k, :)    % m x k, the corresponding indices of the above distances