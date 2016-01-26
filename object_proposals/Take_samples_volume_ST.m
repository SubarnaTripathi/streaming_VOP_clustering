function [ F, fmaps, row, col, all_features ] = Take_samples_volume_ST(img, boxes_per_frame, IoU_mat,X )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
bb_num = size(X,1);

height = size(img{1},1);
width = size(img{1},2);

color_histogram = 1;
resize = 0;

%% fit distribution of only location parameters
onlyLocation = 0; % 0 means with color and location
noLocation = 1;

%%
aspect_ratio_consideration = 1;
canonical_w = 80; 
canonical_h = 80;

if (resize == 0 )
  aspect_ratio_consideration = 0;  
end
%%

all_features = {};

%% histogram feature
binSize = 15;

if color_histogram == 0
    fmaps = zeros(bb_num, binSize);
else
    all_bins = 3*binSize;
    fmaps = zeros(bb_num, all_bins);
end

%%
img_idx_in_volume = 0;

for i = 1: bb_num 
    %% actual frame index
    if ( mod(i-1, boxes_per_frame) == 0 )
        img_idx_in_volume = img_idx_in_volume+1;
        height = size(img{img_idx_in_volume},1);
        width = size(img{img_idx_in_volume},2);
    end
    %%    
    I = imcrop(img{img_idx_in_volume}, uint32(X(i,:)));
    I2 = I;
    
    %% part of features for boxes distribution
    center_y = X(i,1) + (X(i,3)/2); center_x = X(i,2) + (X(i,4)/2);
    area = X(i,3)*X(i,4); n_area = area/(height*width);
    aspect = X(i,4)/X(i,3);
    P = [ center_y/height - 0.5; center_x/width - 0.5; n_area; aspect ]; % location_x; location_y; area; aspect_ratio];
    
    %%
    if (1 == resize)
        resize_w = canonical_w; resize_h = canonical_h;    

        if aspect_ratio_consideration == 1
            w_h_ratio = X(i,2)/X(i,1);
            resize_w = 80; resize_h = 80;    

            %% do some aspect ratio consideration
            if w_h_ratio > 2.5
                resize_w = 200; resize_h = 32;
            elseif w_h_ratio > 1.5
                resize_w = 100; resize_h = 64;
            elseif w_h_ratio < 0.4
                resize_w = 32; resize_h = 200;
            elseif w_h_ratio < 0.67
                resize_w = 64; resize_h = 100;
            end
        end
    
        try
          I2 = imresize(I, [resize_h resize_w]);
        catch
            continue;
        end
    end
    
    
    if color_histogram == 0
        H = imhist(rgb2gray(I2), binSize);
        H = H/sum(H);
        fmaps(i,:) = H'; 
        
        if (1 == onlyLocation)
            %% subarna: normalized feature
            all_features{i} = P;
        else
            %% subarna: normalized feature
            all_features{i} = [H; P];
        end
        
        if (1 == noLocation)
            all_features{i} = [H];
        end
        
        clear H;
    else
        try
            H_R = imhist(I2(:,:,1), binSize);
            H_G = imhist(I2(:,:,2), binSize);
            H_B = imhist(I2(:,:,3), binSize);

            H = [H_R/sum(H_R); H_G/sum(H_G); H_B/sum(H_B)];
            fmaps(i,:) = H';
            
            %% subarna: normalized feature
            if (1 == onlyLocation)
                %% subarna: normalized feature
                all_features{i} = P;
            else
                %% subarna: normalized feature
                all_features{i} = [H; P];
            end
            
            if (1 == noLocation)
                all_features{i} = [H];
            end
                        
            clear H_R H_G H_B H H;
        catch
%             i
%             X(i,:)
        end        
    end
end


%% sampling params
% sig = opts.sig;
% max_offset = 4*sig+1;
% 
% %% sample
% Nsamples = 100;
% sample_from = ones(size(X,1));
% ii = discretesample(sample_from(:)./sum(sample_from(:)),Nsamples);
% ii = unique(ii);
% Nsamples = length(ii);

%F = fmaps;
IoU_thresh = 0.1; % 10%
[row,col] = find(IoU_mat > IoU_thresh);
Nsample = size(row,1);
%keyboard


if color_histogram == 0
    feature_dim = binSize;
else
    feature_dim = all_bins;
end
F = zeros(Nsample, 2*feature_dim);

for i=1:Nsample
    F(i,1:feature_dim) =  fmaps(row(i),:);
    F(i,feature_dim+1: 2*feature_dim) =  fmaps(col(i),:);
end

    
% order A and B so that we only have to model half the space (assumes
%%symmetry: p(A,B) = p(B,A))
F = orderAB(F);

end

