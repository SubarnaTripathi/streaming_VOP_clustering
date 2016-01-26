%%
% Originally a part of: Maggot (developed within EU project CogX)
% Author: Matej Kristan, 2009 (matej.kristan@fri.uni-lj.si; http://vicos.fri.uni-lj.si/matejk/)
% Last revised: 2009
%%
function model = mergeDistributions( model1, model2, mix_weights, testWeights )

if nargin < 4
    testWeights = 1 ;
end

% read dimension and number of components
[ d, N ]= size(model1) ;
 
% augment the model
model.Mu = [ model1.Mu, model2.Mu ] ;
model.Cov = horzcat( model1.Cov, model2.Cov ) ;
model.w = [ model1.w*mix_weights(1), model2.w*mix_weights(2) ] ;

if testWeights == 1
    if ( abs(sum(model.w)-1) > 1e-5 )
        error('Weights should sum to one!!') ;
    end
    model.w = model.w / sum(model.w) ;
end