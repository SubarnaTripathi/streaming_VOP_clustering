function [ F, fmaps, row, col ] = Take_samples_ST(img, IoU_mat,X)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
bb_num = size(X,1);

color_histogram = 1;
resize = 0;

%%
aspect_ratio_consideration = 1;
canonical_w = 80; 
canonical_h = 80;

if (resize == 0 )
  aspect_ratio_consideration = 0;  
end
%%

HOG = 0; %1; 

%% histogram feature
binSize = 15;

if color_histogram == 0
    fmaps = zeros(bb_num, binSize);
else
    all_bins = 3*binSize;
    fmaps = zeros(bb_num, all_bins);
end

for i = 1: bb_num    
    
    %% validity already checked while creating samples
%     if ( X(i,1) < 1 ), X(i,1) = 1; end
%     if ( X(i,2) < 1 ), X(i,2) = 1; end
%     
%     if (X(i,3) < 1 || X(i,4) < 1)
%         continue;
%     end    
%     if (X(i,1)+ X(i,3) > size(img,1)), X(i,3) = size(img,1) - X(i,1); end
%     if (X(i,2)+ X(i,4) > size(img,2)), X(i,4) = size(img,2) - X(i,2); end    
    
    I = imcrop(img, uint32(X(i,:)));
    I2 = I;
    
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
        clear H;
    else

        try
            H_R = imhist(I2(:,:,1), binSize);
            H_G = imhist(I2(:,:,2), binSize);
            H_B = imhist(I2(:,:,3), binSize);

            H = [H_R/sum(H_R); H_G/sum(H_G); H_B/sum(H_B)];
            fmaps(i,:) = H';
        catch
            i
            X(i,:)
        end
        
        clear H_R H_G H_B H;
    end
    %keyboard
end


% if (1 == HOG )
%     %% HOG features
%     binSize = 8;
%     nOrients = 9; % number of orientation bins
%     sz = (uint32(canonical_h/binSize))*(uint32(canonical_w/binSize))*(nOrients*3+5);
%     S = zeros(bb_num, sz);
%     for i =1: bb_num
%         I = single(imcrop(img, X(i,:)))/255;
%         I2 = imresize(I, [canonical_h canonical_w]);
%         H = fhog(I2,binSize,nOrients);
%         fmaps(i,:) = reshape(H, [sz 1]); 
%         %keyboard
%     end
% end
%%%%%%%%%%%%%%%%%


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


%keyboard
    
% order A and B so that we only have to model half the space (assumes
%%symmetry: p(A,B) = p(B,A))
F = orderAB(F);

%keyboard

end

