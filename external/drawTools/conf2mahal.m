% CONF2MAHAL - Translates a confidence interval to a Mahalanobis
%              distance.  Consider a multivariate Gaussian
%              distribution of the form
%
%   p(x) = 1/sqrt((2 * pi)^d * det(C)) * exp((-1/2) * MD(x, m, inv(C)))
%
%              where MD(x, m, P) is the Mahalanobis distance from x
%              to m under P:
%
%                 MD(x, m, P) = (x - m) * P * (x - m)'
%
%              A particular Mahalanobis distance k identifies an
%              ellipsoid centered at the mean of the distribution.
%              The confidence interval associated with this ellipsoid
%              is the probability mass enclosed by it.  Similarly,
%              a particular confidence interval uniquely determines
%              an ellipsoid with a fixed Mahalanobis distance.
%
%              If X is an d dimensional Gaussian-distributed vector,
%              then the Mahalanobis distance of X is distributed
%              according to the Chi-squared distribution with d
%              degrees of freedom.  Thus, the Mahalanobis distance is
%              determined by evaluating the inverse cumulative
%              distribution function of the chi squared distribution
%              up to the confidence value.
%
% Usage:
% 
%   m = conf2mahal(c, d);
%
% Inputs:
%
%   c    - the confidence interval
%   d    - the number of dimensions of the Gaussian distribution
%
% Outputs:
%
%   m    - the Mahalanobis radius of the ellipsoid enclosing the
%          fraction c of the distribution's probability mass
%
% See also: MAHAL2CONF

% Copyright (C) 2002 Mark A. Paskin
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function m = conf2mahal(c, d)

m = chi2inv(c, d);
% pr = 0.95 ; c = (1 - pr)/2 ; 
% m = norminv([c 1-c],0,1) ;
