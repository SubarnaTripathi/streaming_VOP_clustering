% Demo for Edge Boxes (please see readme.txt first).
function out_bbs = VidEdgeBoxes_st(infolder, positive_range, io_mask_all, nFrames, lambda, outfolder)

%addpath(genpath('.'));

%% load pre-trained edge detection model and set opts (see edgesDemo.m)
model=load('models/forest/modelBsds'); model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;

%% set up opts for edgeBoxes (see edgeBoxes.m)
opts = edgeBoxes;
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .75;     % nms threshold for object proposals
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 1e4;  % max number of boxes to detect

%keyboard
numShots = size(positive_range,2);

% load 'test_images1.mat';%load 'test_images.mat';
% nFrames = 48;
%keyboard

%load 'temp_map_50.mat';


%% set up opts for spDetect (see spDetect.m)
% opts = spDetect;
% opts.nThreads = 4;  % number of computation threads
% opts.k = 512;       % controls scale of superpixels (big k -> big sp)
% opts.alpha = .5;    % relative importance of regularity versus data terms
% opts.beta = .9;     % relative importance of edge versus color terms
% opts.merge = 0;     % set to small value to merge nearby superpixels at end

out_bbs = {};

shot_frames = 1;
shot_num = 1;
index_in_shot = 0;

p = io_mask_all{shot_num};
for i= 1:nFrames
    %I = value{i};
    frame_name = sprintf('%s/%08d.jpg', infolder, i);
    I = imread(frame_name, 'jpg');
   
    %% adjust frame_number as per frames per shot
    if (index_in_shot == size(p,1) )
        %% take the last frame's result
        IO = p{index_in_shot};  
        index_in_shot = index_in_shot + 1;    
    else
        if (index_in_shot == size(p,1) + 1 )
            shot_num = shot_num + 1;
            index_in_shot = 0;
        
            if(shot_num > numShots)
                break; %return
            end
        end
        index_in_shot = index_in_shot + 1;     
        if (shot_frames == i)
            p = io_mask_all{shot_num};
            shot_frames = shot_frames + size(p,1)+1;
        end 
        %index in shot
        if (index_in_shot > size(p,1) + 1 )
            break;
        end
        IO = p{index_in_shot}; 
    end
    %%
           
    %IO = accumulated_IO{i};


 %% detect Edge Box bounding box proposals (see edgeBoxes.m)  
 %   tic, bbs = edgeBoxes_ST(I,IO, 0, model,opts); toc  
    tic, bbs_new = edgeBoxes_ST(I,IO, 1, model, lambda, opts); toc 

    %% show evaluation results (using pre-defined or interactive boxes)
    %gt=[15 212 300 100; 141 4 338 340];    
    gt=[15 212 300 100];
    gt(:,5)=0; 
%     [gtRes,dtRes]=bbGt('evalRes',gt,double(bbs),0.5); %subarna: 0.7
%     figure(2); bbGt('showRes',I,gtRes,dtRes(dtRes(:,6)==1,:));
%     title('green=matched gt  red=missed gt  dashed-green=matched detect');
    
%     [gtRes_new,dtRes_new]=bbGt('evalRes',gt,double(bbs_new),0.5); %subarna: 0.7
%     figure(1); bbGt('showRes',I,gtRes_new,dtRes_new(dtRes_new(:,6)==1,:));
%     title('green=matched gt  red=missed gt  dashed-green=matched detect');
%%
    bbsfolder = sprintf('%s/bbs', outfolder);
    if( ~exist( bbsfolder, 'dir' ) ), mkdir( bbsfolder), end;
    fname = sprintf('%s/bbs_new_%d', bbsfolder,i);
    
    save(fname, 'bbs_new');
    
    out_bbs{i} = bbs_new;
%%   
        
    %%    
%     tic, [E, ~,~, seg]=edgesDetect(I,model); toc
%     figure(2); im(1-E);
    
%     cd out; cd edges;
%     myfig = sprintf('my_figure%d',i);
%     print(gcf,'-djpeg',myfig);
%     cd ..;  cd ..; 
    
%     %% detect and display superpixels (see spDetect.m)
%     tic, [S,V] = spDetect(I,E,opts); toc
%     figure(2); im(V);
%     
%     cd out; cd sp;
%     myfig = sprintf('my_figure%d',i);
%     print(gcf,'-djpeg',myfig);
%     cd ..;  cd ..; 
    
    %%
%     %% compute ultrametric contour map from superpixels (see spAffinities.m)
%     tic, [~,~,U]= spAffinities(S,E,segs,opts.nThreads); toc
%     figure(3); im(1-U); 
%     
%     cd out; cd umc;
%     myfig = sprintf('my_figure%d',i);
%     print(gcf,'-djpeg',myfig);
%     cd ..;  cd ..; 
    
end



