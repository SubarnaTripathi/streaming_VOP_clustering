function [X, numSigPoints ] = getAllSigmaPoints( f, k )
% calculates a set of all sigma points for all components and stores
% them in a set of column vectors.

num_components = length(f.weights) ;
dim = rows( f.mu ) ;
numSigPoints = numSigmaPoints( dim, k ) ;
X = zeros( dim, numSigPoints*num_components ) ;
current = 1 ;
for i = 1 : num_components
    P = reshape(f.covariances(i,:),dim,dim) ;
    select = [current:current+numSigPoints-1] ;
    X(:,select) = getSigmaPoints( f.mu(:,i), P, k ) ;
    current = current + numSigPoints ;
end

% ----------------------------------------------------------------------- %
function d = numSigmaPoints( dim, k )
d = 2*dim + (k~=0) ;

% ----------------------------------------------------------------------- %
function X = getSigmaPoints( mu, P, k )
% returns 2n+k sigma points starting with Mu
% as the first point

n = size(P,1) ;
[u,s] = eig(P) ;
S = u*sqrt(s)*sqrt(n+k) ; 
S = [S,-S] ; 

Mu = repmat(mu,1,2*n) ;
X = S+Mu ;
if k ~= 0 X = [mu,X] ; end
