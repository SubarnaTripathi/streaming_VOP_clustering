function demoDistanceGMM()
%
% This is a demo code for the unscented Hellinger distance between a pair of 
% Gaussian mixture models. The code follows the derivation of the
% multivariate unscented Hellinger distance introduced in [1].
% Unlike the Kullback-Leibler divergence, the Hellinger distance is a
% metric between distribution and is constrained to interval (0,1) with 0
% meaning complete similarity and 1 complete dissimilarity.
%
% The demo constructs two random GMMs, displays them, and calculates
% the Hellinger distance between them. This procedure is repeated for 
% one, two, three and one hundred dimensional example.
%
% The demo uses the "drawTools" from Matej Kristan for visualization only.
% If you only want the distance measure, then you will only need the files
% from the folder "uHellingerGMM".
%
% If you are using this distance function in your work, please cite the paper [1].
%
% [1] M. Kristan, A. Leonardis, D. Skoèaj, "Multivariate online Kernel Density
% Estimation", Pattern Recognition, 2011. 
% (url: http://vicos.fri.uni-lj.si/data/publications/KristanPR11.pdf)
% 
% Author: Matej Kristan (matej.kristan@fri.uni-lj.si) 2012

% add path to draw tools (for visualization only)
pth = [pwd, '/drawTools' ] ; rmpath(pth) ; addpath(pth) ;

% add path to unscented Hellinger distance
pth = [pwd, '/uHellingerGMM' ] ; rmpath(pth) ; addpath(pth) ;

figure(1); clf ;
d = [1 2 3 100] ; % dimensionality of data
for i_d = 1:length(d)
    % construct two GMMs
    kde1.pdf = getARandomGMM(d(i_d), 5, 0) ;
    kde2.pdf = getARandomGMM(d(i_d), 3, 1) ;
    
    % display the GMMS
    subplot(1, length(d), i_d) ;
    visualizeKDE('kde', kde1, 'decompose', 0, 'showkdecolor', 'r' ) ; hold on ;
    visualizeKDE('kde', kde2, 'decompose', 0, 'showkdecolor', 'b' ) ;
    
    % get distance between the GMMs
    H = uHellingerJointSupport2_ND( kde1.pdf, kde2.pdf ) ;
    title(sprintf('%dD Hellinger dist: %1.2f', d(i_d), H)) ;
end

% ----------------------------------------------------------------------- %
function pdf_out = getARandomGMM(d,N,delt)

pdf_out.Mu = rand(d,N) + delt ;
pdf_out.w = rand(1,N) ;
pdf_out.w = pdf_out.w/sum(pdf_out.w) ;
pdf_out.Cov = {} ;
for i = 1 : N
    pdf_out.Cov = horzcat(pdf_out.Cov, diag(rand(1,d))) ;
end





