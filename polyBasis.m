function phi_x = polyBasis(x, d)
%POLYBASIS Polynomial basis function
%
% INPUT
% x:       covariate
% d:       polynomial degree
%
% OUTPUT
% phi_x:   [1, x^1, x^2, ... x^d]
    phi_x = bsxfun(@power, x, 0:d);
end

