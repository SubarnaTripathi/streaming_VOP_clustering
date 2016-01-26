function H = uHellingerJointSupport2_ND( f1, f2, varargin )
% Matej Kristan (2008)
% Calculates approximated Hellinger distance between f1 and f2 
% using the unscented transform. Distance gives values on interval [0,1].
% For example, let f1 and f2 be two mixtures of gaussians. Then the
% Hellinger divergence is H2 = 2(1 - int(sqrt(f1*f2) dx)). The Hellinger
% distance is defined as H = sqrt(H2/2), and is related to the well-known
% Bhattacharyya distance.
% 
% Caution: To avoid numerical errors, the distance is thresholded, such
%           that it never falls below zero in "uHell( f1, f2, k )".
% 
%
% Input :
% --------------------------------------------------
% f1    ... first gaussian mixture
% f2    ... second gaussian mixture
% smart ... if this is -1 then distance is calculated from pdf with the
%           least Gaussian components to the distribution with the most
%           components. If the parameter is not set, or its value is 0,
%           then distance from f1 to f2 is calcualted. If the parameter 
%           is set to 1, then then the distribution with most components
%           is used as the first distribution. Even though
%           Hellinger distance is metric, the uHellinger is not entirely
%           symmetric due to approximations of the integrals.
%
% Output :
% --------------------------------------------------
% H     ... square-rooted Hellinger divergence divided by sqrt(2) such that 
%           it takes values from interval [0 1], 0 meaning "the closest" 
%           and 1 meaning "the furthest".
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
 
useWeightedHellinger = 1 ;
useMarginals = 0 ;
featureWeights = [] ;
args = varargin; 
for i = 1 : 2 : length(args)
    switch args{i}        
        case 'useMarginals', useMarginals = args{i+1} ; i = i + 2 ;
        case 'featureWeights', featureWeights = args{i+1} ; i = i + 2 ;    
        case 'useWeightedHellinger', useWeightedHellinger = args{i+1} ; i = i + 2 ;
        otherwise
            msg = sprintf('Unknown switch "%s"!',args{i});
            error(msg) ;
    end
end
 
if useMarginals == 0
    % evaluate direct multidimensional Hellinger
    H = main_uHellinger(f1, f2, useWeightedHellinger) ;
else
    % evaluate Hellinger using marginal approximation
    d = size(f1.Mu,1) ;
    Hm = [] ;
    for i = 1 : d
        f1_i = marginalizeMixture( f1, i ) ;
        f2_i = marginalizeMixture( f2, i ) ;
        Htmp = main_uHellinger(f1_i, f2_i) ;
        Hm = [Hm , Htmp ] ;
    end
    if isempty(featureWeights)
        H = sqrt(sum(Hm.^2)) ;
    else
        H = sum( featureWeights.*Hm ) ;
    end
end

% ----------------------------------------------------------------------- %
function H = main_uHellinger(f1, f2, useWeightedHellinger)

if useWeightedHellinger == 0
    f1.w = f1.w / sum(f1.w) ;
    f2.w = f2.w / sum(f2.w) ;
end

f0 = mergeDistributions( f1, f2, [0.5 0.5], 0 ) ;
f0.w = f0.w / sum(f0.w) ;
 
% remove negative components from proposal
f0 = preprocess_proposal( f0 ) ;

% k = 2 ;1.2;  1.2;1.2; 2; 1.2; 1.5; 2; 1.5 ; %1.1 ;2 ; 
MaxV = 3 ;
[X, sigPointsPerComponent, w, k ] = getAllSigmaPointsOnMixture( f0, MaxV ) ;
 

W = repmat(f0.w,sigPointsPerComponent,1) ;
W = reshape(W,1,length(f0.w)*sigPointsPerComponent) ;
w2 = repmat(w,1,length(f0.w)) ;
W = W.*w2 ;

pdf_f1 = evaluatePointsUnderPdf(f1, X) ;
pdf_f2 = evaluatePointsUnderPdf(f2, X) ;
 
pdf_f1 = pdf_f1.*(pdf_f1 > 0) ;
pdf_f2 = pdf_f2.*(pdf_f2 > 0) ;

pdf_f0 = evaluatePointsUnderPdf( f0, X ) ;
 

g = (sqrt(pdf_f1)- sqrt(pdf_f2)).^2 ;
H = sqrt(abs(sum(W.*g./pdf_f0)/2)) ; 

% g = ((pdf_f1)- (pdf_f2)).^2 ;
% H = sqrt(abs(sum(W.*g./pdf_f0)/4)) ; 

% H = getL2distance( f1, f2 ) ;

% ------------------------------------------------------------------ %
function f0 = preprocess_proposal( f0 ) 

idx_w = find(f0.w > 0) ;
f0.w = f0.w(idx_w) ;
f0.Mu = f0.Mu(:,idx_w) ;
f0.Cov = f0.Cov(idx_w)  ;




 