% Demo for Edge Boxes (please see readme.txt first).

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


%load 'test_images.mat';
load 'test_images1.mat';

nFrames = size(value,2);  %nFrames = size(value,1);
%nFrames = floor(nFrames/20);
nFrames = 48;
%keyboard

load 'temp_map_50.mat';


%% set up opts for spDetect (see spDetect.m)
% opts = spDetect;
% opts.nThreads = 4;  % number of computation threads
% opts.k = 512;       % controls scale of superpixels (big k -> big sp)
% opts.alpha = .5;    % relative importance of regularity versus data terms
% opts.beta = .9;     % relative importance of edge versus color terms
% opts.merge = 0;     % set to small value to merge nearby superpixels at end


for i= 1:nFrames

    %% detect Edge Box bounding box proposals (see edgeBoxes.m)
    I = value{i};
    %I = imread('peppers.png');
    
    IO = accumulated_IO{i};
    
%     imshow(I);
%     pause
    
    tic, bbs = edgeBoxes_ST(I,IO, 0, model,opts); toc  
    tic, bbs_new = edgeBoxes_ST(I,IO, 1, model,opts); toc 

    %% show evaluation results (using pre-defined or interactive boxes)
    %gt=[15 212 300 100; 141 4 338 340];
    gt=[15 212 300 100];
    if(0), 
      gt='Please select an object box.'; disp(gt); figure(1); imshow(I);
      title(gt); [~,gt]=imRectRot('rotate',0); gt=gt.getPos(); 
    end

    gt(:,5)=0; 
    [gtRes,dtRes]=bbGt('evalRes',gt,double(bbs),0.5); %subarna: 0.7
    figure(1); bbGt('showRes',I,gtRes,dtRes(dtRes(:,6)==1,:));
    title('green=matched gt  red=missed gt  dashed-green=matched detect');
    
    [gtRes_new,dtRes_new]=bbGt('evalRes',gt,double(bbs_new),0.5); %subarna: 0.7
    figure(2); bbGt('showRes',I,gtRes_new,dtRes_new(dtRes_new(:,6)==1,:));
    title('green=matched gt  red=missed gt  dashed-green=matched detect');
    
    %keyboard

%     cd out; cd gteval7;
%     myfig = sprintf('my_figure%d',i);
%     print(gcf,'-djpeg',myfig);
%     cd ..;  cd ..;
 

%%
    
    fname = sprintf('out/bbs_new/bbs_new_%d', i);
    save(fname, 'bbs_new');
%%
    
    %pause
  
% 
%     %% show evaluation results (using pre-defined or interactive boxes)
%     gt = [];
% 
%     % gt=[122 248 92 65; 
%     gtRes = [1,1,1,1,1];
%     figure(1); bbGt('showRes',I,gtRes,dtRes(dtRes(:,5)>0.12,:));
%     %figure(1); bbGt('showRes',I,gtRes,dtRes(dtRes(:,6)==1,:));
%     %title('green=matched gt  red=missed gt  dashed-green=matched detect');
%     
%     cd out; cd boxes;
%     myfig = sprintf('my_figure%d',i);
%     print(gcf,'-djpeg',myfig);
%     cd ..;  cd ..;
    
    
    
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


% %% run and evaluate on entire dataset (see boxesData.m and boxesEval.m)
% if(~exist('boxes/VOCdevkit/','dir')), return; end
% 
% split='val'; data=boxesData('split',split);
% nm='EdgeBoxes70'; opts.name=['boxes/' nm '-' split '.mat'];
% edgeBoxes(data.imgs,model,opts); opts.name=[];
% boxesEval('data',data,'names',nm,'thrs',.7,'show',2);
% boxesEval('data',data,'names',nm,'thrs',.5:.05:1,'cnts',1000,'show',3);
