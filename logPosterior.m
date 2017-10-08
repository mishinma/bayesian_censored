function [ logpst ] = logPosterior( phi_x, y, c, w, alpha, beta )
% LOGPOSTERIOR Compute the incomplete data log-posterior
%
% INPUT
% phi_x:    design marix
% y:        survival times
% c:        observed/censored values
% w:        regression coefficients
% alpha:    hyperparameter
% beta:     hyperparameter
%
% OUTPUT
% logpst:   the incomplete data log-posterior

    d = length(w);
    log_pw = -d/2*log(2*pi*1/alpha) - 1/2*(w'*alpha*eye(d)*w);
    sum_log_pz = 0;
    mean_z = phi_x*w;
    for i = 1:length(y)
        if c(i)
            % y_i = c_i
            sum_log_pz = sum_log_pz + ...
                log(1 - normcdf(y(i), mean_z(i), beta^(-1/2)));
        else   
            % y_i < c_i
            sum_log_pz = sum_log_pz + ...
                log(normpdf(y(i), mean_z(i), beta^(-1/2)));
        end 
    end
    logpst = log_pw + sum_log_pz;
end

