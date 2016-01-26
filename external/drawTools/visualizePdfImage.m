function p = visualizePdfImage( datapoints, bounds, grans, sig )
       
minx = bounds(1) ;
maxx = bounds(2) ;

miny = bounds(3) ;
maxy = bounds(4) ;

N = grans ;
x = linspace(minx, maxx, N ) ;
y = linspace(miny, maxy, N ) ;

[X,Y] = meshgrid(x,y) ;
X = X(:)' ;
Y = Y(:)' ;

data = [X;Y] ;
p = zeros(1,length(data)) ;
p = reshape(p,length(y), length(x)) ;
 
for i = 1 : length(datapoints)
    [a, yy] = min(abs(datapoints(2,i) - y)) ;
    [a, xx] = min(abs(datapoints(1,i) - x)) ;
    p(yy,xx) = p(yy,xx) + 1 ;    
end

scl = length(y)/(y(length(y)) - y(1)) ;
 
sig = sig*scl ;
h = fspecial('gaussian', round([1, 1]*(sig*4+1)), sig ) ;
p = imfilter(p, h);
p = flipud(p) ;

% figure(fignum) ; clf ; 
% imagesc(flipud(p)) ; colormap gray ;
 

