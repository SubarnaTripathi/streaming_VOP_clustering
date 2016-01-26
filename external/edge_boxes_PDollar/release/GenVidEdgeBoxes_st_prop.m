% Demo for Edge Boxes (please see readme.txt first).
%function out_bbs = GenVidEdgeBoxes_st(infolder, positive_range, io_mask_all, nFrames, lambda, outfolder)
function [out_bbs, out_bbs_noscore, iou] = GenVidEdgeBoxes_st_prop(infolder, positive_range, io_mask_all, I, lambda, outfolder)

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

% load('test_images1.mat');
% nFrames = 48;

%load('temp_map_50.mat');

%% set up opts for spDetect (see spDetect.m)
% opts = spDetect;
% opts.nThreads = 4;  % number of computation threads
% opts.k = 512;       % controls scale of superpixels (big k -> big sp)
% opts.alpha = .5;    % relative importance of regularity versus data terms
% opts.beta = .9;     % relative importance of edge versus color terms
% opts.merge = 0;     % set to small value to merge nearby superpixels at end

out_bbs = {};
out_bbs_noscore = {};

%%
if (lambda ~= 0 )
    temp_flag = 1;
else
    temp_flag = 0;
end
%%

%% debug
debug_test = 1; %% subarna

nFrames = 1;
%%
for i = 1:nFrames
    %fname = bb_gt(i).im;
    %fname = fname(end-7:end);
    %frame_name = sprintf('%s/%s.jpg', infolder, fname);
    %I = imread(frame_name, 'jpg');
    
    if (iscell(io_mask_all) )
        IO = io_mask_all{1};
    else
        IO = io_mask_all;
    end  %% subarna
       
    %%
    if (debug_test == 1 )
        %% check for size
        orig_I = I;        
        I = imresize(I, [500 500]);
%         if size(I,1)~= size(IO,1) || size(I,2) ~= size(IO,2)
%             %IO = imresize(IO, [size(I,1) size(I,2)]);
%             %I = imresize(I, [500 500]);
%         end
    end

 %% detect Edge Box bounding box proposals (see edgeBoxes.m)  
 %   tic, bbs = edgeBoxes_ST(I,IO, 0, model,opts); toc  
    tic, bbs_new = edgeBoxes_ST(I,IO, temp_flag, model, lambda, opts); toc 
    
    %% show evaluation results (using pre-defined or interactive boxes)
    %gt=[15 212 300 100; 141 4 338 340];    
%     gt=[15 212 300 100];
%     gt(:,5)=0; 
%     [gtRes,dtRes]=bbGt('evalRes',gt,double(bbs),0.5); %subarna: 0.7
%     figure(2); bbGt('showRes',I,gtRes,dtRes(dtRes(:,6)==1,:));
%     title('green=matched gt  red=missed gt  dashed-green=matched detect');
    
%     [gtRes_new,dtRes_new]=bbGt('evalRes',gt,double(bbs_new),0.5); %subarna: 0.7
%     figure(1); bbGt('showRes',I,gtRes_new,dtRes_new(dtRes_new(:,6)==1,:));
%     title('green=matched gt  red=missed gt  dashed-green=matched detect');
%%


%     gt = bb_gtTraining(i).boxes;
%     gt(:,5) = 0;
%     [oa]= bbGt('compOas',gt,double(bbs_new)); %subarna: 0.7
%     
% 
%     bbsfolder = sprintf('%s/bbs', outfolder);
%     if( ~exist( bbsfolder, 'dir' ) ), mkdir( bbsfolder), end;
%     fname = sprintf('%s/bbs_%s', bbsfolder,fname);
%     
%     save(fname, 'bbs_new');

    %% subarna
    if ( debug_test == 1 )
        if size(orig_I,1)~= size(I,1) || size(orig_I,2) ~= size(I,2)
            %% to-left corner location of the box
            bbs_new(:,1) =  (bbs_new(:,1)-1)*size(orig_I,2)/size(I,2) + 1;
            bbs_new(:,2) =  (bbs_new(:,2)-1)*size(orig_I,1)/size(I,1) + 1; 
            
            %% width and height of the boxes
            bbs_new(:,3) =  bbs_new(:,3)*size(orig_I,2)/size(I,2);            
            bbs_new(:,4) =  bbs_new(:,4)*size(orig_I,1)/size(I,1); 
        end
    end
    %%

    
    out_bbs{i} = bbs_new;    
    num_proposals = min(2000, size(bbs_new,1));
    out_bbs_noscore{i} = bbs_new(1:num_proposals, 1:4);
    iou = {};  %% subarna
    %iou{i} = oa(1:num_proposals);
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



