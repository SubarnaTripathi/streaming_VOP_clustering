function pdf = evaluateDistributionAt( mu, weights, covariances, locations )
% mu          ... mean values of mixture components
% weights     ... weights of mixture components
% covariances ... covariance s of components
% locations   ... points where distribution is to be evaluated

dim = sqrt(double(cols(covariances))) ;
num_data = cols(double(mu)) ;
num_locations = cols(double(locations)) ;
[precisions, determinants] = getPrecisionsAndDets( double(covariances) ) ;

pi_2d = (2.0*pi)^double(dim) ;
constantsA = double(weights)./sqrt(pi_2d*double(determinants)) ;

pdf = double(zeros(1,num_locations)) ;
for i_data = 1 : num_data
    point = mu(:,i_data) ;
    Precision_i = reshape(precisions(i_data,:),dim,dim) ;
    D_2 = double(sqdist(locations,point,Precision_i)) ; 
    pdf = pdf + constantsA(i_data).*exp(-0.5*D_2) ;
end