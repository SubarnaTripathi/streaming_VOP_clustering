function kde_out = constructKDE(data, weights, H, Covs)
% constructs the kde as a Gaussian mixture model 

if nargin < 4
    Covs = [] ;
end

if isempty(weights) 
    weights = ones(1,size(data,2))/size(data,2) ;
end

pdf.Mu = data ;
pdf.w = weights ;
C = {} ;
for i = 1 : size(data,2)
    if isempty(Covs)
        C = horzcat(C, H) ;  
    else    
        C = horzcat(C, H+Covs{i}) ;
    end
end
pdf.Cov = C ;
kde_out.pdf = pdf ;