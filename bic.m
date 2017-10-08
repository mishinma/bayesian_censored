function [ crt ] = bic( logPost, d, nObs )
% BIC   Compute the Bayesian Information Criterion (BIC)
% 
% INPUT
% logPost:     data log posterior
% d:           degree of the polinomial
% nObs:        number of obersvation
%
% OUTPUT
% crt:         BIC criterion 
    crt = (d+1)*log(nObs) - 2*logPost;
end

