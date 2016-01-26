% Demo of crisp boundaries
%
% -------------------------------------------------------------------------
% Crisp Boundaries Toolbox
% Phillip Isola, 2014 [phillpi@mit.edu]
% Please email me if you find bugs, or have suggestions or questions
% -------------------------------------------------------------------------


%% setup
% first cd to the directory democontaining this file, then run:
%compile; % this will cEheck to make sure everything is compiled properly; if it is not, will try to compile it


%% Detect boundaries
% you can control the speed/accuracy tradeoff by setting 'type' to one of the values below
% for more control, feel free to play with the parameters in setEnvironment.m

%load 'test_images.mat';
load 'test_images1.mat';

%type = 'super_speedy'; % use this for fastest results
type = 'speedy'; % use this for slightly slower but crisper results
%type = 'accurate'; % use this for slow, but high quality results

nFrames = size(value,2);  %nFrames = size(value,1);
nFrames = floor(nFrames/20);
%keyboard

for i= 1:nFrames

    %I = imread('test_images/101027.jpg');
    I = value{i};
    [E,E_oriented] = findBoundaries(I,type);

    close all; subplot(121); imshow(I); subplot(122); imshow(1-mat2gray(E));
    
    keyboard
    
    cd out; cd edge;
    myfig = sprintf('my_figure%d',i);
    print(gcf,'-djpeg',myfig);
    cd ..; cd ..;

    %keyboard

        
    %% Segment image
    % builds an Ultrametric Contour Map from the detected boundaries (E_oriented)
    % then segments image based on this map
    %
    % this part of the code is only supported on Mac and Linux

    if (~ispc)

        thresh = 0.1; % larger values give fewer segments
        E_ucm = contours2ucm_crisp_boundaries(E_oriented,type);
        S = ucm2colorsegs(E_ucm,I,thresh);

        close all; subplot(121); imshow(I); subplot(122); imshow(S);
        
        cd out; cd segment;
        myfig = sprintf('my_figure%d',i);
        print(gcf,'-djpeg',myfig);
        cd ..; cd ..;
    end

end
