function [X, numSigPoints, w, k ] = getAllSigmaPointsOnMixture( f, MaxV, varargin ) %, prev_neg_k ) %_OldWorking
% calculates a set of all sigma points for all components and stores
% them in a set of column vectors.

% if nargin < 3
%     prev_neg_k = 1 ;
% end      

useGeneralizedSigma = 0 ;
prev_neg_k = 1 ;
% process arguments
args = varargin;
nargs = length(args);
for i = 1:2:nargs
    switch args{i}        
        case 'useGenSigmaPoints', useGeneralizedSigma = args{i+1} ;          
        otherwise
            error('unknown parameter %s', args{i}) ;
    end
end


num_components = length(f.w) ;
dim = rows( f.Mu ) ;

n = dim ;
k = MaxV - n ;


% prevent negative weights
if prev_neg_k == 1 & k < 0 
    k = 0 ; 
    MaxV = k + n ;
end

% k=1 ;

numSigPoints = numSigmaPoints( dim, k ) ;
X = zeros( dim, numSigPoints*num_components ) ;
current = 1 ;
for i = 1 : num_components 
    select = [current:current+numSigPoints-1] ;
    X(:,select) = getSigmaPoints( f.Mu(:,i), f.Cov{i}, k ) ;
    current = current + numSigPoints ;
end
 
if k == 0
    wk = [] ;
else
    wk = k / (n+k) ;
end   
w = [ wk, ones(1,2*n)*(1/(2*(n+k)))] ;
minTol = 1e-5 ;
if abs(sum(w) - 1) > minTol
%     w = w / sum(w) ;
    error('Weights in the unscented transform should sum to one!') ;
end
 
 




% ----------------------------------------------------------------------- %
function d = numSigmaPoints( dim, k )
d = 2*dim + (k~=0) ;

% ----------------------------------------------------------------------- %
function X = getSigmaPoints( mu, P, k, gener )
% returns 2n+k sigma points starting with Mu
% as the first point

n = size(P,1) ;
% [u,s] = eig(P) ;
[u,s,v] = svd(P) ;
S = u*sqrt(s)*sqrt(n+k) ; 
% S = [S,-S] ; 
S = reshape([S;-S],n,n*2) ;

Mu = repmat(mu,1,2*n) ;
 

X = S+Mu ;
if k ~= 0 X = [mu,X] ; end
