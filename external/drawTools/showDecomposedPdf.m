%%
% Originally a part of: Maggot (developed within EU project CogX)
% Author: Matej Kristan, 2009 (matej.kristan@fri.uni-lj.si; http://vicos.fri.uni-lj.si/matejk/)
% Last revised: 2009
%%
function returnBounds = showDecomposedPdf( pdf, varargin )
%
% 'linTypeSum'  ... line type of summed pdf
% 'linTypeSub'  ... line type of components
% 
%
draw_to_these_axes = [] ;
applySummation = 1 ;
linTypeSum = 'r' ;
linTypeSub = 'k' ;
decompose = 1 ;
returnBounds = 0 ;
bounds = [] ;
showDashed = 0 ;
priorWeight = 1 ;
enumComps = 0 ;
args = varargin;
nargs = length(args);
for i=1:2:nargs
    switch args{i}
        case 'bounds', bounds = args{i+1} ;
        case 'showDashed', showDashed = args{i+1} ; 
        case 'priorWeight', priorWeight = args{i+1} ;
        case 'returnBounds', returnBounds = args{i+1} ;
        case 'decompose', decompose = args{i+1} ;
        case 'linTypeSum', linTypeSum = args{i+1} ;
        case 'linTypeSub', linTypeSub = args{i+1} ;
        case 'enumComps', enumComps = args{i+1} ;
        case 'applySummation', applySummation = args{i+1} ;
        case 'draw_to_these_axes', draw_to_these_axes = args{i+1} ;
    end
end
 

if isempty(draw_to_these_axes)
   draw_to_these_axes = gca ; 
end

if showDashed == 1 
    linTypeSum = [linTypeSum(1),'--'] ;
elseif showDashed == 0 
    linTypeSum = [linTypeSum] ; 
end

if isempty(bounds)
    b1 = max(sqrt(max((pdf.Cov{1})))) ;
    bmin = min([pdf.Mu]) - b1*5 ;
    bmax = max([pdf.Mu]) + b1*5 ;
    bounds = [bmin,bmax] ;
end

if returnBounds > 0 && nargout > 0
    returnBounds = bounds ;
else
    returnBounds = [] ;
end

pdf.w = priorWeight*pdf.w ;

if decompose == 1
    h = get(draw_to_these_axes,'NextPlot') ;     
    for i = 1 : length(pdf.w)
        if i == 2 
            set(draw_to_these_axes,'NextPlot','add') ;
        end
        showPdf( bounds, 1000, pdf.Mu(:,i), pdf.Cov{i}, pdf.w(i), linTypeSub, 2, draw_to_these_axes) ;
        if ( enumComps == 1 )
           text(pdf.Mu(:,i), 5, num2str(i)) ;%, 'FontSize',16
        end
    end
    if applySummation == 1
        showPdf( bounds, 1000, pdf.Mu, cell2mat(pdf.Cov)', pdf.w, linTypeSum,2, draw_to_these_axes  ) ;
    end    
%     if ( h == 0 ) hold off ; end
    set(draw_to_these_axes, 'NextPlot', h) ;
else
    showPdf( bounds, 1000, pdf.Mu, cell2mat(pdf.Cov)', pdf.w, linTypeSum, 2, draw_to_these_axes  ) ;
end
 
% ca = axis ;
%axis([bounds,0,ca(4)]);
axis tight

% ----------------------------------------------------------------------- %
function showCurrentResult(f_ref, data, fignum)
figure(fignum); clf; 
dat_scale = 1 ;

b1 = sqrt(max([cell2mat{f_ref.covariances}])) ;
bmin = min([f_ref.Mu]) - b1*5 ;
bmax = max([f_ref.Mu]) + b1*5 ;
bounds = [bmin,bmax] ;

subplot(1,3,1) ; hold on ;
[hst, hst_x] = hist(data,[-1:0.1:0]) ; hst = hst / sum(hst) ; bar(hst_x,hst) ; axis tight ; ca=axis;
line([0,0],[0,ca(4)],'color','g');
axis([[-1,1]*dat_scale,0,ca(4)]);

subplot(1,3,2) ; 
pdf_pos = constructKDEfromData( data, 'compression', 0 ) ;
for i = 1 : length(pdf_pos.w)
   showPdf( bounds, 1000, pdf_pos.Mu(:,i), pdf_pos.Cov{i}, pdf_pos.w(i), 'k', 1) ; 
end
showPdf( bounds, 1000, pdf_pos.Mu, pdf_pos.covariances, pdf_pos.w, 'r',2  ) ;
ca = axis ;
line([0,0],[0,ca(4)],'color','g');   
axis([[-1,1]*dat_scale,0,ca(4)]);

subplot(1,3,3) ; hold on ;
for i = 1 : length(f_ref.w)
   showPdf( bounds, 1000, f_ref.Mu(:,i), f_ref.Cov{i}, f_ref.w(i), 'k',1 ) ; 
end 
showPdf( bounds, 1000, f_ref.Mu, f_ref.covariances, f_ref.w, 'r',2  ) ;
ca = axis ;
line([0,0],[0,ca(4)],'color','g');   
axis([[-1,1]*dat_scale,0,ca(4)]);
 
drawnow ;
% ----------------------------------------------------------------------- %
function y_evals = showPdf( bounds, N,centers, covariances, w, color, lw, draw_to_these_axes )
x_evals = [bounds(1):abs(diff(bounds))/N:bounds(2)] ;
y_evals = evaluateDistributionAt( centers, w, covariances, x_evals ) ;
plot ( draw_to_these_axes, x_evals, y_evals, 'Color', color, 'LineWidth',lw ) ;
 
% ---------------------------------------------------------------------- %
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

% --------------------------------------------------------------------- %
function [precisions, determinants] = getPrecisionsAndDets( covariances )
num_points = rows(covariances) ; 
dim = sqrt(cols(covariances)) ;
len_dim = dim^2 ;

precisions = double(zeros( num_points, len_dim )) ;
determinants = double(zeros( 1, num_points )) ;
for i_point = 1 : num_points
    Covariance = reshape(covariances(i_point,:),dim,dim ) ;
    Precision = inv(Covariance) ;
    precisions(i_point,:) = reshape(Precision, 1, len_dim ) ;
    determinants(i_point) = abs(det(Covariance)) ;
end


