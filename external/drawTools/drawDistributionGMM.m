%%
% Originally a part of: Maggot (developed within EU project CogX)
% Author: Matej Kristan, 2009 (matej.kristan@fri.uni-lj.si; http://vicos.fri.uni-lj.si/matejk/)
% Last revised: 2009
% --------------------
% Example: drawDistributionGMM( 'pdf', pdf ) ;
% switch for color: 'color' T
% By default, it draws in nidm 2 sigma bound (~95%). To change it 
% sigma, use: 'sigmaPlotBound' 
% 'decompose'  shows also components
% drawDistributionGMM( 'pdf', pdf, 'sigmaPlotBound', 3 ) ; draws 3 sigma bound
%
%%
function drawDistributionGMM( varargin )

draw_to_these_axes = [] ;
maxAlpha = 0.7 ;
useEdgeColorBlack = 1 ;
deactivateFaceColor = 0 ;
useAlphaWeights = 1 ;
weighted = 0 ;
sigmaPlotBound = 2 ;
decompose = 0 ;
pdf = [] ;
% process arguments
color = 'b' ;
args = varargin;
nargs = length(args);
for i = 1:2:nargs
    switch args{i}        
        case 'pdf', pdf = args{i+1} ; 
        case 'color', color = args{i+1} ; 
        case 'decompose', decompose = args{i+1} ; 
        case 'sigmaPlotBound', sigmaPlotBound = args{i+1} ;
        case 'weighted', weighted = args{i+1} ;
        case 'useAlphaWeights', useAlphaWeights = args{i+1} ;
        case 'deactivateFaceColor', deactivateFaceColor = args{i+1} ;
        case 'useEdgeColorBlack', useEdgeColorBlack =  args{i+1} ;
        case 'draw_to_these_axes', draw_to_these_axes =  args{i+1} ;
    end
end

if isempty(pdf)
    return ;
end

if isempty(draw_to_these_axes)
    draw_to_these_axes = gca ;
end

% read dimension and number of components
[ d, N ]= size(pdf.Mu) ;
if d > 3
    warning('Can"t draw larger than 3 dim! Exiting draw function...');
    return ;
end
 
if ( d == 1 )
    if ( ~isempty(color) )
        showDecomposedPdf( pdf, 'linTypeSum', color, 'decompose', decompose, 'draw_to_these_axes', draw_to_these_axes ) ;
    else
        showDecomposedPdf( pdf, 'decompose', decompose, 'draw_to_these_axes', draw_to_these_axes ) ;
    end

    return ;

end

if useAlphaWeights == 1
    EdgeAlphaVal = ones(1,length(pdf.w)) ;
    for i = 1 : length(pdf.w)
%          pdf.Cov{i} =  pdf.Cov{i} + eye(size( pdf.Cov{i}))*max( diag(pdf.Cov{i}) )*1e-5 ;
 
        EdgeAlphaVal(i) = pdf.w(i)*normpdfmy(pdf.Mu(:,i), pdf.Cov{i}, pdf.Mu(:,i)) ; 
 
    end    
    denom = sum(EdgeAlphaVal) ;
    if denom == 0 denom = 1 ; end        
    EdgeAlphaVal = maxAlpha*( EdgeAlphaVal / denom ) ;
else
    EdgeAlphaVal = ones(1,length(pdf.w))*0.2 ;
end

if useEdgeColorBlack == 1
    EdgeColor = [0 0 0] ;
else
    EdgeColor = color ;
end

h = get(draw_to_these_axes,'NextPlot') ;
FaceColor = color ;
for i = 1 : N
%     if weighted == 1
%        color 
%     end
    
    if i == 2 
        set(draw_to_these_axes,'NextPlot','add') ;
        %hold on ;
    end

    switch d
        case 1
%              plotgauss1d( pdf.Mu(i), pdf.Cov{i}, pdf.w(i) ) ;               
        case 2
             plotgauss2d( pdf.Mu(:,i), pdf.Cov{i}, color, sigmaPlotBound, draw_to_these_axes ) ;
        case 3            
            if deactivateFaceColor == 1
                    FaceColor = 'none' ;
            end
            soptds = {'EdgeAlpha', 0.1, 'FaceAlpha', EdgeAlphaVal(i), 'FaceColor',...
                        FaceColor, 'EdgeColor', EdgeColor };  %
            plotcov3(pdf.Mu(:,i), pdf.Cov{i}  , draw_to_these_axes,  'surf-opts', soptds) ;
            box on; grid on ;
    end 
end
%  if ( h == 0 ) hold off ; end    
set(draw_to_these_axes,'NextPlot',h);
