% Censored Bayesian polynomial regression
% Project work for the course 
% CS-E4820 - Machine Learning: Advanced Probabilistic Methods,
% Aalto University, 2017
% This script is based on the template provided by the course staff

clear all; close all; clc;
    
% Set parameters
alpha = 1; % Prior precision
beta = 1; % Noise precision
maxDegree = 5; % We consider polynomials with degree 1,...,max_degree
nIter = 30; % Number of iterations of the EM algorithm
nStarts = 10; % Number of random starts of EM

% Load data
% x: covariate, y: survival time, 
% c: c(i)=0 -> y(i) observed value, c(i)=1 -> y(i) censored value
load project_data.mat;
nObs = length(y);

% Initialize data structures for storing results
bicVals= zeros(maxDegree, 1);
wVals = cell(maxDegree, 1);

for d = 1:maxDegree
    
    bestLogPost = -Inf;
    best_w = -Inf*ones(d+1, 1); % MAP estimate for parameters
    phi_x = polyBasis(x, d);
    S = inv(beta*(phi_x'*phi_x) + alpha*eye(d+1));
    E_z = zeros(size(y));
    
    for s = 1:nStarts
        % Initialize weights
        w = mvnrnd(zeros(1,d+1), alpha^(-1/2)*eye(d+1))'; 
        % Incomplete data log-posterior
        logPosts = zeros(nIter + 1, 1); 
        % Log-posterior before first iteration
        logPosts(1) = logPosterior(phi_x, y, c, w, alpha, beta);     
        
        for i = 1:nIter
            % E-step  
            mean_z = phi_x*w;
            for j = 1:nObs
                if c(j)
                    alpha2 = (y(j) - mean_z(j))/beta^(-1/2);
                    E_z(j) = mean_z(j) + beta^(-1/2)*H_function(alpha2);
                else 
                    E_z(j) = y(j);
                end
            end
            % M-step
            w = beta*S*phi_x'*E_z;
        
            % Compute log-posterior
            logPosts(i+1) = logPosterior(phi_x, y, c, w, alpha, beta); 
        end
        if logPosts(nIter+1) > bestLogPost
            bestLogPost =  logPosts(nIter+1);
            best_w = w;
            logPostconvergences = logPosts;
        end
        
    end
    
    % Plotting
    xLim = 2;
    xVals = -1*xLim:0.1:xLim;
    yVals = (polyBasis(xVals', d)*best_w)'; 
    
    comp_w = beta*S*phi_x'*y; % Fit a regression model for y
    comp_y = (polyBasis(xVals', d)*comp_w)';
        
    figure;
    subplot(2, 1, 1);
    hold on;
    scatter(x(~c), y(~c));
    scatter(x(c), y(c), 'Marker', '*');
    plot(xVals, yVals);
    plot(xVals, comp_y, 'Color', 'b');
    title(['Polynomial of degree ' num2str(d)]);
    legend('Observed', 'Censored', 'Censored regression', ... 
           'Standard regression', 'Location', 'southeast');
    hold off;
    
    subplot(2, 1, 2);
    plot(0:nIter, logPostconvergences);
    title('Log-posterior');
    
    % Compute BIC
    bicVals(d) = bic(bestLogPost, d, nObs); 
    
    % Store MAP estimate
    wVals{d} = best_w;
end
figure;
bar(bicVals);
title('Bayesian Information Criterion (BIC)');
xlabel('Degree of polynomial');