
function [seg] = generate_volume_segmentations(rectangle, image, obj_num, boxes_per_frame)
    %UNTITLED Summary of this function goes here
    %   Detailed explanation goes here     
    mask = zeros(size(image,1), size(image, 2));
    num_rect = size(rectangle,1);
    %% simple weight coefficient
    wgt_coeff = 1/num_rect;  %
    
    %threshold = max(0.015, num_rect*0.15*0.001); %% at least 15% voting or a hard threshold 
    threshold = 0.1; %0.15
    maxIterations = 20;
    
    %% test : debug
%     wgt_coeff = 0.1; 
%     threshold = num_rect*0.15*wgt_coeff; %at least 15% voting or a hard threshold 
    %%
    
    fprintf('segmenting object number %d\n', obj_num);
    
    for rect_idx = 1: num_rect
        row1 = rectangle(rect_idx,2); row2 = rectangle(rect_idx,2) + rectangle(rect_idx,4);
        col1 = rectangle(rect_idx,1); col2 = rectangle(rect_idx,1) + rectangle(rect_idx,3);
        mask(row1:row2, col1:col2) = mask(row1:row2, col1:col2)+wgt_coeff;
    end


    seg = compute(image,mask,threshold,maxIterations);
    
end


function seg = compute(image,mask,threshold,maxIterations)
% For one image only.

% settings
    doMorph = false; % apply morphological operations at the end (as in original GrabCut)
    doVisualize = false; % show plots with intermediate results

    if size(image,1)~=size(mask,1) || size(image,2)~=size(mask,2)
        mask = img_resize(mask,[size(image,1) size(image,2)]);
    end;

    % output
    segs = cell(1,maxIterations+1);
    flows = zeros(1,maxIterations+1);
    energies = zeros(1,maxIterations+1);
    converged = false;

    % initialization
    [img, h, w] = getImage(image);
    [P, Pk] = getPairwise(img);
    [fgm, bgm] = initializeModel(img,mask>=threshold);
    U_loc = cat(3,mask,1-mask);

    for i = 1:maxIterations

    %	pg_message('iteration %d/%d',i,maxIterations);

        [fgk bgk] = assignComponents(img,fgm,bgm);
        U_app = getUnary_app(img,fgm,bgm,fgk,bgk);
        U = getUnary(U_loc,U_app);

        [segs{i} flows(i) energies(i)] = getSegmentation(P,U,w,h);

    %	pg_message('flow = %g, energy = %g',flows(i),energies(i));

        % TODO assert energy/flow decrease

        if doVisualize
            visualize(img,mask,threshold,segs{i},i,energies(1:i));
        end;

        if i>1 && all(segs{i-1}(:)==segs{i}(:))
    %		pg_message('converged after %d/%d iterations',i,maxIterations);
            converged = true;
            break;
        end;

        [fgm bgm] = learnModel(img,segs{i},fgm,bgm,fgk,bgk);

    end;

    if ~converged
    %	pg_message('did not converge after %d iterations',maxIterations);
        fprintf('did not converge after %d iterations\n',maxIterations);
    end;

    segs = segs(1:i);
    flows = flows(1:i);
    energies = energies(1:i);

    seg = segs{end};
    energy = energies(end);

    if doMorph
        seg = applyMorph(seg);
        if doVisualize
            visualize(img,mask,threshold,boxes,seg,energies);
        end;
    end;

end


function energy = getEnergy(A,T,labels)

    energy = 0;
    energy = energy + sum(T(labels==0,2));
    energy = energy + sum(T(labels==1,1));
    energy = energy + sum(sum(A(labels==0,labels==1)));
end


function [img h w] = getImage(img)

    img = im2double(img);
    assert(ndims(img)==3);

    h = size(img,1);
    w = size(img,2);
    assert(size(img,3)==3);
end


function [fg bg] = initializeModel(img,mask)

    % pg_message('initializeModel');

    assert(any(mask(:)));
    assert(any(~mask(:)));

    img = reshape(img,[],3);

    K = 5;

    fg = pdf_gm.fit_using_vectorquantisation(img(mask,:),K);
    bg = pdf_gm.fit_using_vectorquantisation(img(~mask,:),K);
end


function [fk bk] = assignComponents(img,fg,bg)

    % pg_message('assignComponents');

    fk = fg.cluster_2d(img);
    bk = bg.cluster_2d(img);
end


function [fg bg] = learnModel(img,seg,fg,bg,fk,bk)

    % pg_message('learnModel');

    K = 5;

    img = reshape(img,[],3);
    seg = reshape(seg,[],1);
    fk = reshape(fk,[],1);
    bk = reshape(bk,[],1);

    fg = pdf_gm.fit_given_labels(img(seg,:),fk(seg),K,fg);
    bg = pdf_gm.fit_given_labels(img(~seg,:),bk(~seg),K,bg);
end


function [A K] = getPairwise(img)

    % pg_message('getPairwise');

    [h,w,~] = size(img);
    n = h*w;

    imgr = img(:,:,1); imgr = imgr(:);
    imgg = img(:,:,2); imgg = imgg(:);
    imgb = img(:,:,3); imgb = imgb(:);

    % locations
    [x,y] = meshgrid(1:w,1:h);
    x = x(:); y = y(:);

    % neighbors down -> y+1 -> idx+1
    n1_i1 = 1:n; n1_i1 = n1_i1(y<h);
    n1_i2 = n1_i1+1;

    % neighbors right-down -> x+1,y+1 -> idx+1+h
    n2_i1 = 1:n; n2_i1 = n2_i1(y<h & x<w);
    n2_i2 = n2_i1+1+h;

    % neighbors right -> x+1 -> idx+h
    n3_i1 = 1:n; n3_i1 = n3_i1(x<w);
    n3_i2 = n3_i1+h;

    % neighbors right-up -> x+1,y-1 -> idx+h-1
    n4_i1 = 1:n; n4_i1 = n4_i1(x<w & h>1);
    n4_i2 = n4_i1+h-1;

    from = [n1_i1 n2_i1 n3_i1 n4_i1];
    to = [n1_i2 n2_i2 n3_i2 n4_i2];

    gamma = 50; % TODO could be trained
    invdis = 1./sqrt((x(from)-x(to)).^2+(y(from)-y(to)).^2);
    dz2 = (imgr(from)-imgr(to)).^2 + (imgg(from)-imgg(to)).^2 + (imgb(from)-imgb(to)).^2;
    beta = (2*mean(dz2.*invdis))^-1; % TODO changed, .*invdis is not in paper, but in gco
    expb = exp(-beta*dz2);
    c = gamma * invdis .* expb;

    A = sparse([from to],[to from],[c c]); % TODO do i need to explicitely make it symmetric?

    K = 1+max(sum(A,2)); % TODO changed, gco seems to have only half of this, not correct
end


function T = getUnary_app(img,fg,bg,fk,bk)

% pg_message('getUnary');

    T = cat(3,fg.pdf_2d(img,fk),bg.pdf_2d(img,bk));
end


function U = getUnary(U_loc,U_app)

    U = -log(U_loc .* U_app);
    U = reshape(U,[],2);
    U = sparse(U);
end


function [seg, flow, energy] = getSegmentation(P,U,w,h)

    [flow, labels] = maxflow(50*P,50*U);
    seg = reshape(labels==1,h,w);
    energy = getEnergy(P,U,labels);
end


function visualize(img,mask,threshold,seg,iteration,energies)

    clf();

    % subplot2d(5,1,1,1);
    % plot_image(img,'image');
    % 
    % subplot2d(5,2,2,1);
    % plot_image(mask,'prior');
    % 
    % subplot2d(5,2,2,2);
    % plot_image(mask>=threshold,'init');
    % 
    % subplot2d(5,1,3,1);
    % plot_image(img.*repmat(double(seg),[1 1 3]),'foreground');
    % 
    % subplot2d(5,1,4,1);
    % plot_image(seg,'segmentation');
    % 
    % subplot2d(5,1,5,1);
    % plot(energies);

    subplot(2,3,1);
    imshow(img); %'image'

    subplot(2,3,2);
    imshow(mask,[]); % ,'prior'

    subplot(2,3,3);
    imshow(mask>=threshold); %,'init'

    subplot(2,3,4);
    imshow(img.*repmat(double(seg),[1 1 3])); %,'foreground'

    subplot(2,3,5);
    imshow(seg);%,'segmentation'

    subplot(2,3,6);
    plot(energies);

    title('convergence');
    ylabel('energy');
    xlabel('iteration');

    drawnow();
    
    %keyboard
end


function seg = applyMorph(seg)
    seg = imclose(seg,strel('disk',3));
    seg = imfill(seg,'holes');
    seg = bwmorph(seg,'open'); % remove thin regions
    [~,N] = bwlabel(seg); % select largest 8-connected region
    h = hist(seg(:),1:N);
    [~,i] = max(h);
    seg = (seg==i);
end