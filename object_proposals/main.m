%function images = main ()
function main ()

   %% debug_mode = 1 => write all intermediate results in files, intermediate matrices as images
    debug_mode = 0;
    
    %enable grab-cut segmentation
    segment = 1;
    
    opts = setEnvironment('speedy');
    
    %% enable automatic cluster number generation (slow version)
    auto_cluster_number = 1;

    %% video: cat and bird from youtube
    frame_numbers = [2 5 8 10 15 20 25 30 35 40 45 48];
    %frame_numbers = [48];
    disp('...loading video data');
    all_images = load('test_images1.mat', '-mat');
    images = all_images.value;
    disp('video data loaded ...');
    %%
    h = waitbar(0,'Starting object proposal analysis frame-wise...');
    
    %% test
    only_GOP = 0;
    only_EB = 1;
    if (1 == only_EB)
        only_GOP = 0;
    end
    %%
    eb_thresh = 0.25; % threshold for edge_box score for being valid object proposal
    
    for img_num = frame_numbers
        waitbar(img_num/size(frame_numbers,2));
        
        %[ img, X1, X2 ] = read_bbs (images,img_num,eb_thresh);
        [ img, X1, X2 ] = read_bbs_new (images,img_num,eb_thresh);
        
        %% test/debug
        %%only GOP
        if (1 == only_GOP )
            t = size(X2,1);
            temp = X2;
        elseif (1 == only_EB)
            t = size(X1,1);
            temp = X1;
        end
        clear X1 X2;
        offset = uint32(t/2);
        X1 = temp(1:offset, :);
        X2 = temp(offset+1:t, :);
        clear temp;
        
        [ IoU_mat ] = calc_iou_mat( img, X1, X2 );
       
        %% save input image
        dirname = sprintf('bird_cat/inputs');    
        if (~exist(dirname,'dir' )), mkdir(dirname), end        
        im_name = sprintf('%s/input%d.png', dirname, img_num);
        imwrite(img,im_name,'png'); 
        
        X = [X1; X2];
        
        [Containment, A_ratio, As_diff] = calc_containment_ratio_mat(X);                
        [F, f_maps, row, col] = Take_samples_ST(img, IoU_mat, X);
        
        %figure(100), imshow(IoU_mat),title('IoU matrix');

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
            
            for i=1:2 % for some re1ason repeatedly running fminsearch continues to improve the objective
                bw = getBW(p,1);
                bw_star = fminsearch(@(bw) f(bw,p,F_val), bw(1:size(bw,1)/2), fminsearch_opts);
                bw_star = cat(1,bw_star,bw_star);
                adjustBW(p,min(max(bw_star,reg_min),reg_max));
            end
        end
        
        %% learn w predictor
        rf = learnPMIPredictor_ST(f_maps,p,opts, row, col); %% [];        
        affinity = buildW_pmi_ST(f_maps,rf,p,opts,row, col);   
                

        %% spectralPb_fast_custom
        nvec = 2;          
        orig_sz = [size(X1,1)+ size(X2,1), 1]; 
        
        %%
        affinity_new = full(affinity); 
        %figure(101), imshow(affinity_new, []),title('affinity');
        
        %% write affinity matrix as image files
        dirname = sprintf('bird_cat/results/affinity');      
        if (~exist(dirname,'dir' )), mkdir(dirname), end  
        
        if (1 == debug_mode )
            im_name = sprintf('%s/orig_%d.png', dirname, img_num);
            imwrite(affinity_new/sum(sum(affinity_new)),im_name,'png');
        end
        
        affinity_new = update_affinity(affinity_new, Containment, A_ratio, IoU_mat, As_diff);
        %figure(102), imshow(affinity_new, []),title('updated affinity');
        
        if (1 == debug_mode )
            im_name = sprintf('%s/updated_%d.png', dirname, img_num);
            imwrite(affinity_new/sum(sum(affinity_new)),im_name,'png');
        end
        

        
        %try
            if(0 == auto_cluster_number)
                %[E, E_oriented, vect] = find_vect(affinity_new, orig_sz, nvec); %%subarna  
                [~, ~, vect] = find_vect(affinity_new, orig_sz, nvec); %%subarna  
                [all_rects, obj_num] = cluster(vect, nvec, X1, X2, img, img_num, 0);
            else            
                %% ZP clustering
                [clusts_R, rBestGroupIndex, ~, ~] = cluster_rotate(affinity_new, [1 2 3 4 5]);
               
                idx = zeros(size(X,1),1);
                obj_num = length(clusts_R{rBestGroupIndex});
                for l = 1: obj_num
                    temp_cols = clusts_R{rBestGroupIndex}{l};
                    %all_rects{l} = X(temp_cols,:);                    
                    idx(temp_cols) = l;
                end
                %% write visualization code
                [all_rects] = save_visualization_result(idx, obj_num, X, img, img_num, 0);
                
                clear idx;
            end                     
        %catch
        %end
        
        %keyboard
        
        if (1 == segment)
            try
                dirname = sprintf('bird_cat/results/out_segments/%d', img_num);
                if (~exist(dirname,'dir' )), mkdir(dirname), end 
                
                for bb_idx=1:obj_num
                    rectangle = uint32(all_rects{bb_idx});
                    seg = generate_segmentations(rectangle, img, bb_idx);
            
                    %%% save segmentation output                   
                    im_name = sprintf('%s/outsegment_%d.png', dirname, bb_idx);
                    imwrite(seg,im_name,'png'); 

                    clear seg rectangle;
                end
            catch             
              
            end
        end
        
        %keyboard
        
        clear X1 X2 X IoU_mat IoU_mat_e img BW Label;
        clear Containment A_ratio As_diff;
        clear affinity_new;
        clear all_rects;
    
    end    
    clear images windows;
        
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

function [all_rects, obj_num] = cluster(vect, nvec, dt1, dt2, I, im_num, exponentiated )
    vec(1:size(vect,1),1:nvec) = vect(:,1,:);
    
    bb = [dt1; dt2];
        
    cluster_num = nvec; %nvec;
    obj_num = cluster_num;
    
    %idx3 = kmeans(vec,cluster_num);
    idx = kmeanspp(vec',cluster_num);
    
    [all_rects] = save_visualization_result(idx, cluster_num, bb, I, im_num, exponentiated);
    
    %% debug
%     J1 = I;    
%     all_rects = [];    
%     randcolors = rand([cluster_num,3]);    
%     for i = 1:cluster_num
%         colorname = uint8(255.*randcolors(i,:)); % [R G B]; 
%         shapeinname = vision.ShapeInserter('Shape','Rectangles','BorderColor','Custom','CustomBorderColor',colorname);
%         
%         index = find(idx == i);
%         rectangle = int32(bb(index,:));
%         
%         all_rects{i} = rectangle;
%         
%         %%
%         J = step(shapeinname, I, rectangle);
%         save_results_debug(1, exponentiated, im_num, J, i);
% 
%         %% : debug
%         J1 = step(shapeinname, J1, rectangle);   
%       
%         clear J;               
%     end       
%     save_results_debug(2, exponentiated, im_num, J1, i);
%     clear J1;
       
    obj_num = cluster_num;
end


function save_results_debug(variation, exponentiated, im_num, J,i)

%% variation 1: means save individual clusters in different image
%% variation 2: means save all the clusters in single image
switch(variation)
    
    case 1 
        %% save results
        if (exponentiated == 1)
            dirname = sprintf('bird_cat/results_exponentiated/%d',im_num);
        else
            dirname = sprintf('bird_cat/results/%d',im_num);
        end
        
        if (~exist(dirname,'dir' )), mkdir(dirname), end        
        im_name = sprintf('%s/%d.png', dirname, i);
        imwrite(J,im_name,'png');
        
    case 2
         %% save results
        if (exponentiated == 1)
            dirname = sprintf('bird_cat/results_exponentiated');
        else
            dirname = sprintf('bird_cat/results');
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


function [all_rects] = save_visualization_result(idx, cluster_num, bb, I, im_num, exponentiated)
    
    J1 = I;   
    
    all_rects = [];
    
    randcolors = rand([cluster_num,3]);    
    for i = 1:cluster_num
        colorname = uint8(255.*randcolors(i,:)); % [R G B]; 
        shapeinname = vision.ShapeInserter('Shape','Rectangles','BorderColor','Custom','CustomBorderColor',colorname);
        
        index = find(idx == i);
        rectangle = int32(bb(index,:));
        
        all_rects{i} = rectangle;
        
        %%
        J = step(shapeinname, I, rectangle);
        save_results_debug(1, exponentiated, im_num, J, i);

        %% : debug
        J1 = step(shapeinname, J1, rectangle);   
      
        clear J;
               
    end
       
    save_results_debug(2, exponentiated, im_num, J1, i);
    clear J1;
end

