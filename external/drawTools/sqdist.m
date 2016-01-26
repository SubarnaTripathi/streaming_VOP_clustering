function d = sqdist(centrs, point, precision)

dA = bsxfun(@minus, centrs, point)' ; 
A = dA * precision ;
d = sum(A.*dA,2)' ;