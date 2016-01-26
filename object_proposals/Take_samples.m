function [ F ] = Take_samples(img, IoU_mat,X, D, opts)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
bb_num = size(X,1);
canonical_w = 64;
canonical_h = 64;

HOG = 0; %1; 

%% histogram feature
binSize = 10;
fmaps = zeros(bb_num, binSize);
for i = 1: bb_num
    I = imcrop(img, X(i,:));
    if ( X(i,1) < 1 ), X(i,1) = 1; end
    if ( X(i,2) < 1 ), X(i,2) = 1; end
    if ( X(i,3) < 1 || X(i,4) < 1)
        continue;
    end
    
    I2 = imresize(I, [canonical_h canonical_w]);

    H = imhist(rgb2gray(I2), binSize);
    H = H/sum(H);
    fmaps(i,:) = H; 
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
sig = opts.sig;
max_offset = 4*sig+1;

%% sample
Nsamples = 100;
sample_from = ones(size(X,1));
ii = discretesample(sample_from(:)./sum(sample_from(:)),Nsamples);
ii = unique(ii);
Nsamples = length(ii);

F = fmaps;

%%
    % choose random offset
    r = randn(Nsamples,1)*sqrt(sig);
    r_n = r./sqrt(sum(r.^2));
    r = r+r_n;
    
    % cap offset
    s = sign(r);
    r = min(abs(r),max_offset);
    r = s.*r;
    
    % make pair of points
    p1 = ii;
    p2 = ii;
%     p1(:) = p1(:)+r;
%     p2(:) = p0(:)-r;
%     p1 = round(p1);
%     p2 = round(p2);
%     
%     % remove out of bounds samples
%     m = (p1(:,1)<1) | (p1(:,2)<1) | (p2(:,1)<1) | (p2(:,2)<1) | ...
%         (p1(:,1)>im_size(1)) | (p1(:,2)>im_size(2)) | (p2(:,1)>im_size(1)) | (p2(:,2)>im_size(2));
%     p1 = p1(~m,:);
%     p2 = p2(~m,:);


    %
%     F = zeros(size(X1,1),size(fmaps,2));  %(size(P1,1),size(f_maps,3)*2)
%     for c=1:size(fmaps,2) 
%         tmp = fmaps(:,c);
%         F(:,c) = tmp(ii);
%     end
    
    %% order A and B so that we only have to model half the space (assumes
    % symmetry: p(A,B) = p(B,A))
    %F = orderAB(F);

end

