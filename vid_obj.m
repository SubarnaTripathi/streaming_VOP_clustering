%% video object localization


%% compile mex files for temporal edge
addpath( genpath( '.' ) );
% cd '/home/subarna/my_code/external/FastVideoSegment/';
% compile;
% cd '/home/subarna/my_code';

%%
options.segment = 0;  %enable grab-cut segmentation
options.debug = 0;  %% enable debug
options.nFrame = [2 5 8 10 51];
options.eb_thresh = 0.25; %threshold for edge_box score for being valid object proposal
%% enable automatic cluster number generation (slow version)
options.auto_cluster = 1;

options.volume_cluster = 1;
options.volume_frames = 3; % default volume is 5 frames (150-189 bb for each subVolume)
options.volumeGrabCut = 0;

%% parameter for spatial and temporal edge combination
lambda = 0.8; %% lambda*motion_edge + (1-lambda)*spatial_edge

options.GMM = 0; %1 = model each cluster distribution using GMM for association; otherwise use KDD

%% do motion analysis first, and create inout mask
foldername = '/home/subarna/my_code/';

all_videos = {'bird_cat'; 'alaskan_bear'; 'baby_and_macaw'; 'bird_cat2'; 'cacatuas'; 'elephant_rescue2'; 'horse_ride'};

for v = 1 : size(all_videos,1)
    
    video_name = all_videos{v}; 
    fprintf('\nstarting processing video file %s\n', video_name);

    %% index to first frame corresponding to each shot
    range = [ 1, 50]; %range = [ 1, 50, 100, 150, 200];
    positive_range = [1]; %positive_range = [ 1, 2, 3, 4 ];
    io_mask = temporal_edge(foldername, video_name, range, positive_range ); %% video_rapid segmentation (intermediate result)
    options.range = range;

    %% Now, call Piotr Dollar's edgeBox code 
    infolder = fullfile( foldername, 'Data', 'inputs', video_name );
    outfolder = fullfile( foldername, 'Data', 'outputs', video_name );
    nFrames = 200;
    
%     % debug: visualize
%     for i=1:max(range)-2
%         a = io_mask{1}{i};
%         figure(1), imshow(a, []); title(sprintf('frame %d', i));
%         figure(2), imshow(a); title(sprintf('frame %d', i));
%         figure(3), imagesc(a); pause; title(sprintf('frame %d', i));
%     end
    %%
    
    bbs = VidEdgeBoxes_st(infolder, positive_range, io_mask, nFrames, lambda, outfolder);

    %% finally use my clustering and segmentations on proposals 
    proposals_clustering(infolder, outfolder, bbs, options);
    
    %keyboard
end

rmpath(genpath('.'));



