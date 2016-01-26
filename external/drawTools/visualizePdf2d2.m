function output = visualizePdf2d2( pdf, bounds, dat_in, grans )

if isfield(pdf,'suffStat') && ~isempty(pdf.suffStat.B)
    output = subspacePrewhitenTransform( 'pdf', pdf, 'minEigenEnergy', 1e-3,  'transDirection', 'forward', 'allLayers', 0 ) ;
else
   output.pdf = pdf ; 
end
    
if size(output.pdf.Mu,1) < 2
    warning('Will not plot due to singularity.') ;
    return ;
end
 
% get negative sigmapoints 
[sigmapoints, ~, ~, ~ ] = getAllSigmaPointsOnMixture( pdf, 3 ) ;

if ~isempty(bounds)
    minx = bounds(1) ; maxx = bounds(2) ;
    miny = bounds(3) ; maxy = bounds(4) ;
else
    minx = min(sigmapoints(1,:)) ; maxx = max(sigmapoints(1,:)) ;
    miny = min(sigmapoints(2,:)) ; maxy = max(sigmapoints(2,:)) ;
    
    c = ([minx+maxx ; miny+maxy])/2 ;
    w = 1.5*([maxx-minx ; maxy-miny])/2 ;
    minx = c(1)-w(1) ; maxx = c(1)+w(1) ;
    miny = c(2)-w(2) ; maxy = c(2)+w(2) ;
    
end
dex = maxx - minx + 1 ;
dey = maxy - miny + 1 ;
scl = dex/dey ;

N = grans ;
if scl < 1
    Nx = round(N*scl) ;
    Ny = N ;
else
    Nx = N ;
    Ny = round(N/scl) ;
end 

x = linspace(minx, maxx, Nx ) ;
y = linspace(miny, maxy, Ny ) ;

[X,Y] = meshgrid(x,y) ;
X = X(:)' ;
Y = Y(:)' ;

data = [X;Y] ;
p = evaluatePointsUnderPdf(pdf, data) ;
p = reshape(p,length(y), length(x)) ;

% figure(fignum) ; clf ; 
if nargout > 0
    output = p ; %flipud(p) ;
else
    imagesc((p)) ; colormap gray ;
    output = p ; %[] ;
end

if nargin == 3  && ~isempty(dat_in)
    dat_in(1,:) = (N-1)*(dat_in(1,:)-minx)/(maxx-minx) + 1 ;
    dat_in(2,:) = (N-1)*(dat_in(2,:)-miny)/(maxy-miny) + 1 ;
    hold on ;
    plot(dat_in(1,:), dat_in(2,:), 'r*') ;
    hold off ;
else
%     dex = maxx - minx ;
%     dey = maxy - miny ;
%     [sy,sx]=size(p) ;
%     scl = dex/dey ;
%     
%     p = imresize(p, [sy,sx*scl]) ;
end

