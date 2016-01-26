%%
% Originally a part of: Maggot (developed within EU project CogX)
% Author: Matej Kristan, 2009 (matej.kristan@fri.uni-lj.si; http://vicos.fri.uni-lj.si/matejk/)
% Last revised: 2009
%%
function [ p, model ] = normpdfmy(Mu, Cov, X, input_minerr)
% % % % % % % % % % % % % % % % % % % % % % %
% Mu, Cov ... gaussian model
% X     ... points
% % % % % % % % % % % % % % % % % % % % % % % 
 
minerr = [] ;
if nargin == 4
    minerr = input_minerr ;
end

model.Mu = Mu ;
model.Cov = {Cov} ;
model.w = 1 ;
[ p, model ] = normmixpdf( model, X, minerr ) ;

if nargout == 1 
    model = [] ; 
end