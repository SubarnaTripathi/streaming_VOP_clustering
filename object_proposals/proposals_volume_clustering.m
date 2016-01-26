function proposals_volume_clustering (infolder, outfolder, bbs, options)
    
%% options for my clustering and segmentation algo
    debug_mode = options.debug;
    segment = options.segment;
    eb_thresh = options.eb_thresh;
    auto_cluster = options.auto_cluster;
    %frame_numbers = options.nFrame;
    
    %% opts for joint kernel estimation and PMI method (as in crisp boundary)
    opts = setEnvironment('speedy');

    frame_numbers = max(options.range);
    volume_frames = options.volume_frames;
    
    h = waitbar(0,'Starting object proposal analysis frame-wise...');
    
    %% only in case of first sub-volume, there is no history
    first_subVolume = 1; 
    universe = {};
    
    universe.auto_cluster = options.auto_cluster;
    universe.GMM = options.GMM; %1 means GMM, otherwise KDD
    
    %%
    for img_num = 1:volume_frames:frame_numbers
        waitbar(img_num/frame_numbers);
        
        if (first_subVolume == 1)
            fprintf('clustering for frame numbers (%d to %d) \n', img_num, img_num+volume_frames-1);
        else
            fprintf('clustering for frame numbers (%d to %d) \n', img_num-1, img_num+volume_frames-1);
        end
        
        [ img, X, boxes_per_frame ] = my_read_volume_bbs(infolder,bbs, img_num,volume_frames, first_subVolume, eb_thresh );
                
        %%
        t = size(X,1);   
        offset = uint32(t/2);
        dt1 = X(1:offset, :);
        dt2 = X(offset+1:t, :);
        
        [ IoU_mat ] = calc_iou_volume_mat( img, dt1, dt2 );
        
        %X = [X1; X2];
        
        [Containment, A_ratio, As_diff] = calc_containment_ratio_mat(X);                
        [F, f_maps, row, col, all_features] = Take_samples_volume_ST(img, boxes_per_frame, IoU_mat, X);
        
        %%
        universe.all_features = all_features; 
        clear all_features;
        
        %% subarna
        %figure(100), imshow(IoU_mat),title('IoU matrix');
        %opts.kde.learn_bw = 1;
        
        %% fit model
        if (~opts.kde.learn_bw)
            p = kde(F',0.05,[],'e');
        else
            p = kde(F','lcv',[],'e');

            %f = @(bw,p) nLOO_LL_anis(bw,p);
            f = @(bw,p,F_val) nLOO_LL_anis(bw,p,F_val);
            fminsearch_opts.Display = 'off';%'iter';
            fminsearch_opts.MaxIter = 20;

            reg_min = opts.kde.min_bw; % this regularizes for things like perceptual discriminability, that do not show up in the likelihood fit
                                       %  reduces the impact of
                                       %  outlier channels and noise
            reg_max = opts.kde.max_bw;
            
            for i=1:2 % for some reason repeatedly running fminsearch continues to improve the objective
                bw = getBW(p,1);
                bw_star = fminsearch(@(bw) f(bw,p,F_val), bw(1:size(bw,1)/2), fminsearch_opts);
                bw_star = cat(1,bw_star,bw_star);
                adjustBW(p,min(max(bw_star,reg_min),reg_max));
            end
        end
        options.auto_cluster = 0;
        %% learn w predictor
        rf = learnPMIPredictor_ST(f_maps,p,opts, row, col); %% [];        
        affinity = buildW_pmi_ST(f_maps,rf,p,opts,row, col);   
                
        %% spectralPb_fast_custom
        nvec = 2;          
        orig_sz = [size(X,1), 1]; 
        
        %%
        affinity_new = full(affinity); 
        %figure(101), imshow(affinity_new, []),title('affinity');
        
        %% write affinity matrix as image files
        
        if (debug_mode == 1)
            dirname = sprintf('%s/affinity', outfolder);      
            if (~exist(dirname,'dir' )), mkdir(dirname), end 
            affinity_file = sprintf('%s/affinity_%d.png', dirname, img_num);
            imwrite(affinity_new, affinity_file);
        end
                
        affinity_new = update_affinity(affinity_new, Containment, A_ratio, IoU_mat, As_diff);
        
        if (debug_mode == 1)
            %figure(102), imshow(affinity_new, []),title('updated affinity');
            affinity_file = sprintf('%s/updated_affinity_%d.png', dirname, img_num);
            imwrite(affinity_new, affinity_file);
        end
        
        %% spectral clustering
        try
            if(0 == auto_cluster)
                %[E, E_oriented, vect] = find_vect(affinity_new, orig_sz, nvec); %%subarna  
                [~, ~, vect] = find_vect(affinity_new, orig_sz, nvec); %%subarna  
                [all_rects, obj_num, all_colors, distributions] = cluster_volume(outfolder,vect, nvec, dt1, dt2, img, img_num, boxes_per_frame, 0, first_subVolume, universe);
         
            else            
                %% ZP clustering
                [clusts_R, rBestGroupIndex, ~, ~] = cluster_rotate(affinity_new, [1 2 3]);
               
                idx = zeros(size(X,1),1);
                obj_num = length(clusts_R{rBestGroupIndex});
                for l = 1: obj_num
                    temp_cols = clusts_R{rBestGroupIndex}{l};
                    %all_rects{l} = X(temp_cols,:);                    
                    idx(temp_cols) = l;
                end
                %% write visualization code
                [all_rects, all_colors, distributions] = save_visualization_result(outfolder,idx, obj_num, X, img, img_num, volume_frames,boxes_per_frame, 0, first_subVolume, universe);
                
                clear idx;
                
            end 
            %% subarna: double check
            %% delete old universe distribution, new distributions already calculated; assign in the next step
            if (1 ~= first_subVolume )
                for i = 1:obj_num 
                    clear universe.distributions{i};
                end
            end
            
            %% dump clusters for debug
            %rectangle = [];
%             for img_idx = 1:volume_frames                
%                  for bb_idx=1:obj_num                    
%                     rectangle = uint32(all_rects{img_idx,bb_idx});                   
%                 
%                     box_mat_name = sprintf('bbox_obj%d_img%d.mat', bb_idx, img_idx);
%                     save(box_mat_name, 'rectangle');
%                 end
%             end
%             keyboard
            %% end debug
            
            %% save for future
            universe.object_numbers = obj_num;
            for obj_idx = 1: obj_num
                universe.all_colors{obj_idx} = all_colors{obj_idx};
                universe.distributions{obj_idx} = distributions{obj_idx};
            end            
        catch
        end        
              
        if (first_subVolume == 1), 
            start_frame_idx = 1;
        else
            start_frame_idx = 2;
        end        
        volume = size(img,2); 
        
        
        %% : debug
        %% single bounding box from the cluster
%         for img_idx = start_frame_idx : volume
%             for bb_idx=1:obj_num                    
%                 try
%                     rectangle = uint32(all_rects{img_idx,bb_idx});
%                     sR = uint32(median(rectangle,1)); 
%                     randcolors = rand([1,3]);  
%                     colorname = uint8(255.*randcolors); % [R G B];
%                     shape_name = vision.ShapeInserter('Shape','Rectangles','BorderColor','Custom','CustomBorderColor',colorname);        
%                     dIm_o = step(shape_name, img{img_idx}, rectangle);
%                     dIm = step(shape_name, img{img_idx}, sR);
%                     figure(1000), imshow(dIm_o);
%                     figure(2000), imshow(dIm); keyboard
%                 catch
%                 end
%             end
%         end
        %%      
        
        if (1 == segment)                        
            if (options.volumeGrabCut == 1 )
                fprintf('VolumeGrabCut option is not yet implemented\n'); 
                fprintf('Setting grabCut on individual frames\n');
            end

            volume = size(img,2);
                
            for img_idx = start_frame_idx : volume
                %% folder name
                if (first_subVolume == 1), 
                    dirname = sprintf('%s/out_segments/%d', outfolder,(img_num-1)+img_idx);
                else
                    dirname = sprintf('%s/out_segments/%d', outfolder,(img_num-2)+img_idx);
                end
                
                if (~exist(dirname,'dir' )), mkdir(dirname), end 

                for bb_idx=1:obj_num                    
                    try
                        rectangle = uint32(all_rects{img_idx,bb_idx});

                        %% for volume case, it might so happen that a cluster number can be present in a subVolume, but not in a particular frame
                        if (size(rectangle,1) < 1), break; end;

                            seg = generate_volume_segmentations(rectangle, img{img_idx}, bb_idx, boxes_per_frame);
                            %%% save segmentation output                   
                            im_name = sprintf('%s/segment_%d.png', dirname, bb_idx);
                            imwrite(seg,im_name,'png'); 

                            clear seg rectangle;                    
                    catch
                    end
                end
            end
        end
        
        clear X1 X2 X IoU_mat IoU_mat_e img BW Label;
        clear Containment A_ratio As_diff;
        clear affinity_new;
        clear all_rects;
        
        %% all except first sub-volume
        if (first_subVolume == 1), first_subVolume = 0; end
    
    end    
    clear images windows;
    clear universe.all_colors universe.kernels universe;    
    close(h)
end

function [E, E_oriented, vect] = find_vect(IoU_mat, orig_sz, nvec)
    try
            [E_oriented, vect] = spectralPb_fast_custom_ST(IoU_mat, orig_sz, nvec);        
            %E_oriented = borderSuppress(E_oriented);
            E = max(E_oriented,[],3); 
    catch
        return
    end
end

function [all_rects, obj_num, all_colors, distributions] = cluster_volume(outfolder,vect, nvec, dt1, dt2, I, im_num, boxes_per_frame, exponentiated, first_subVolume, universe )
    vec(1:size(vect,1),1:nvec) = vect(:,1,:);
    
    bb = [dt1; dt2];
        
    cluster_num = nvec; %nvec;
    obj_num = cluster_num;
    
    %idx3 = kmeans(vec,cluster_num);
    idx = kmeanspp(vec',cluster_num);
    
    volume_frames = max(size(I,2));
    
    [all_rects, all_colors, distributions] = save_visualization_result(outfolder,idx, cluster_num, bb, I, im_num, volume_frames, boxes_per_frame, exponentiated, first_subVolume, universe);
       
    obj_num = cluster_num;
end

function [all_rects, all_colors, distributions] = save_visualization_result(outfolder,idx, cluster_num, bb, I, im_num, volume_frames, boxes_per_frame, exponentiated, first_subVolume, universe)
    
    J1 = I;       
    all_rects = [];
    
    randcolors = rand([cluster_num,3]);  
    
    distributions = {};
    
    %% distinguish between first and other sub-volumes
    if (first_subVolume == 1)
        start_frame_idx = 1;
    else
        start_frame_idx = 2;
    end
    
    %% associate cluster colors from history, and if required create new color    
    for i = 1:cluster_num  
        index = find(idx == i);
        %%
        F = zeros(length(index), size(universe.all_features{1},1));
        for i_temp=1:length(index)
            F(i_temp,:) = universe.all_features{index(i_temp)};
        end
        
        if (universe.GMM == 1 )
            %% fit a GMM 
            num_mix = 2;
            GMModel = fitgmdist(F', num_mix); 
            distributions{i} = GMModel;
        else
            distributions{i} = kde(F',0.05,[],'e');
        end
        %%
        
        if (first_subVolume == 1)
            colorname = uint8(255.*randcolors(i,:)); % [R G B];
        else
            %% associate old color numbers; put actual association code (GMM distance) here
            colorname = associate_last_cluster_or_new(universe, distributions{i});
            %colorname = universe.all_colors{i};
        end
        shapeinname = vision.ShapeInserter('Shape','Rectangles','BorderColor','Custom','CustomBorderColor',colorname);        
        all_colors{i} = colorname;        
        
        %%
        for img_idx = start_frame_idx:volume_frames
            lb = (img_idx-1)*boxes_per_frame; ub = img_idx*boxes_per_frame;
            
            if(universe.auto_cluster)
                [i_temp, ~] = find( (index > lb) & (index <= ub)  );  
            else
                [~,i_temp] = find( (index > lb) & (index <= ub)  ); 
            end
            
            index_temp = index(i_temp);
                        
            rectangle = int32(bb(index_temp,:));
            all_rects{img_idx,i} = rectangle;
        
            %%
            J = step(shapeinname, I{img_idx}, rectangle);
            save_results_debug(outfolder,1, exponentiated, im_num + (img_idx-1), J, i);

            %% : debug
            J1{img_idx} = step(shapeinname, J1{img_idx}, rectangle);   
      
            clear J{img_idx};
        end
        clear J;
               
    end
    
    
    %% test
%         for img_idx = start_frame_idx : volume_frames
%             save_results_debug(outfolder,2, exponentiated, im_num + (img_idx -start_frame_idx), J1{img_idx}, 0); %% second arg = 2 => last arg useless
%             clear J1{img_idx}; 
%         end
        
   
        %%
        for img_idx = start_frame_idx : volume_frames
            save_results_debug(outfolder,2, exponentiated, im_num + (img_idx -start_frame_idx), J1{img_idx}, 0); %% second arg = 2 => last arg useless
            clear J1{img_idx}; 
        end
    
        clear J1{1};

    clear J1;
end


function save_results_debug(outfolder,variation, exponentiated, im_num, J,i)

%% variation 1: means save individual clusters in different image
%% variation 2: means save all the clusters in single image
switch(variation)
    
    case 1 
        %% save results
        if (exponentiated == 1)
            dirname = sprintf('%s/results_exponentiated/per_frame/%d',outfolder,im_num);
        else
            dirname = sprintf('%s/per_frame/%d',outfolder,im_num);
        end
        
        if (~exist(dirname,'dir' )), mkdir(dirname), end        
        im_name = sprintf('%s/%d.png', dirname, i);
        imwrite(J,im_name,'png');
        
    case 2
         %% save results
        if (exponentiated == 1)
            dirname = sprintf('%s/results_exponentiated', outfolder);
        else
            dirname = sprintf('%s/results', outfolder);
        end

        if (~exist(dirname,'dir' )), mkdir(dirname), end        
        im_name = sprintf('%s/clustered%d.png', dirname, im_num);
        imwrite(J,im_name,'png'); 
        
end

end

function [Containment, A_ratio, As_diff] = calc_containment_ratio_mat(X)

bb_num = size(X,1);

A_ratio = zeros(bb_num, bb_num);
A_ratio_temp = zeros(bb_num, bb_num);
Containment = zeros(bb_num, bb_num);
As_diff_temp = zeros(bb_num, bb_num);
As_diff = zeros(bb_num, bb_num);

for i=1:bb_num
    Ai = X(i,3)*X(i,4);
    Aspect_ratio_i = X(i,3)/X(i,4);
    
    for j = 1:bb_num
        Aj = X(j,3)*X(j,4);
        A_ratio_temp(i,j) = Ai/Aj;
        Aspect_ratio_j = X(j,3)/X(j,4);
        
        As_diff_temp(i,j) = abs(Aspect_ratio_j - Aspect_ratio_i);
        
        c = 0;
        
        if ( (X(j,1)> X(i,1))  && (X(j,1)+X(j,3) < X(i,1)+X(i,3)) && (X(j,2)>X(i,2))   &&  (X(j,2)+X(j,4)<X(i,2)+X(i,4)) )
            c = 1;
        end
        Containment(i,j)= c;
    end
end

As_diff(:,:) = max(As_diff_temp, As_diff_temp');
A_ratio(:,:) = min(A_ratio_temp, A_ratio_temp');
Containment = Containment + Containment';

clear A_ratio_temp As_diff_temp;

end


function affinity_new = update_affinity(affinity, Containment, A_ratio, IoU_mat, As_diff)

exponentiate = 0;

A_thresh_low = 0.001; %% low threshold of ratio ratio
A_thresh_high = 0.7; %% high threshold of ratio ratio

As_diff_thresh = 0.75; %% threshold of difference in aspect ratio

%Weight_mat = Containment.*A_ratio;
I = (IoU_mat > 0 );
C = (Containment == 1);
A = (A_ratio <= A_thresh_high);
B = (A_ratio >= A_thresh_low);
D = (As_diff >= As_diff_thresh);
AA = A&B;
CC = C.*AA;
DD = I; 
%DD = DD.*~D;


affinity_new = affinity.*(~CC);
affinity_new = affinity_new.*(DD);

if 1 == exponentiate
    affinity_new = exp(affinity_new);
end

end



function [ img, X, boxes_per_frame] = my_read_volume_bbs(infolder, bbs, img_num, volume_frames, first_frame, eb_thresh )

    max_box_num = 30;
    
    boxes_per_frame = max_box_num;

    X = [];
    img = {};
    
    img_idx_in_volume = 1;
    
    if (first_frame == 1 )
        start_frame = img_num; end_frame = img_num+volume_frames-1;
    else
        start_frame = img_num-1; end_frame = img_num+volume_frames-1;
    end
    
    for frame_num = start_frame : end_frame
        max_frame_num  = size(bbs,2);
        if (max_frame_num < frame_num )
            fprintf('bounding box for current frame is not available\n');
            break;
        end
        
        image_file = sprintf('%s/%08d.jpg', infolder, frame_num);
        if exist(image_file, 'file') 
            img{img_idx_in_volume} = imread(image_file, 'jpg');
        else
            fprintf('warning: no more image file exists');
            break;
        end
                
        img_idx_in_volume = img_idx_in_volume+1;
        
        
        cell_boxes = bbs{frame_num};    
        Y = cell_boxes(:,1:4);

        %% limit bb number

        if(size(Y,1) > max_box_num )
             temp_Y = Y(1:max_box_num, :);
             clear Y; Y = temp_Y;
        end
        
        X = [X; Y];
        
    end
    
end
