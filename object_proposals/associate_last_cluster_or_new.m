%% association of any of the old existing cluster or make a new cluster

function colorname = associate_last_cluster_or_new(universe, distribution)

    dist_thresh = 50; %% check
    %% check pair-wise distance between current kernel and all the other kernels in last clustering
    [dist, t] = calc_all_dist(universe.GMM, distribution, universe.distributions);
    [d, index] = min(dist);      
    
    dist_thresh = max(t); %t(index);
    
    if (d <= dist_thresh)
        colorname = universe.all_colors{index};
    else
        randcolors = rand([1,3]);
        colorname = uint8(255.*randcolors(1,:)); % [R G B];
    end
end


function [dist, t] = calc_all_dist(GMM, current_distribution, history_distributions)
 %%
    last_clust_num = size(history_distributions,2); %% currently always it is 2
    dist = zeros(last_clust_num,1);
    
    %% GMM not supported now
     if (GMM == 1 )
         disp('GMM not supported now');
         exit
%         % construct two GMMs
%         gmm1.pdf = current_distribution;
%         
%         for i=1:last_clust_num
%             gmm2.pdf = history_distributions{i};
% 
%             % display the GMMS
%             %subplot(1, length(d), i_d) ;
%             visualizeKDE('kde', gmm1, 'decompose', 0, 'showkdecolor', 'r' ) ; hold on ;
%             visualizeKDE('kde', gmm2, 'decompose', 0, 'showkdecolor', 'b' ) ;
% 
%             % get distance between the GMMs
%             H = uHellingerJointSupport2_ND( gmm1.pdf, gmm1.pdf ) ;
%             title(sprintf('%dD Hellinger dist: %1.2f', H)) ;
%             keyboard
%         end
    end
    
    for i=1:last_clust_num
        [dist(i), t(i)] = calc_dist(current_distribution, history_distributions{i});
    end

end

function [dist, t] = calc_dist(current_kernel, kernel)
    visualize = 0;
    %% distance between two kernels
    %% KL-divergence
    %evaluate(X,X,'lvout')
%     p = evaluate(current_kernel,current_kernel,'lvout');
%     q = evaluate(kernel,kernel,'lvout');
    
%% evaluate both the kernels on the same positions (corresponding to current_kernel); then compare the distributions
    %% select kde pos based on which ever was defined on less number of points
    if ( getNpts(current_kernel) < getNpts(kernel) )
        kde_pos = current_kernel;
        t = getNpts(current_kernel)*0.5;
    else
        kde_pos = kernel;
        t = getNpts(kernel)*0.5;
    end
    
    p = evaluate(current_kernel,kde_pos);
    q = evaluate(kernel,kde_pos);
        
    dist = KLDiv(p, q);
    
    if (1 == visualize)
        figure(10), subplot(2,1,1), plot(p);
        subplot(2,1,2), plot(q);
    end
end

%http://www.mathworks.com/matlabcentral/fileexchange/20688-Kullback%E2%80%93Leibler-divergence/content/KLDiv.m
function dist=KLDiv(P,Q)
%  dist = KLDiv(P,Q) Kullback-Leibler divergence of two discrete probability
%  distributions
%  P and Q  are automatically normalised to have the sum of one on rows
% have the length of one at each 
% P =  n x nbins
% Q =  1 x nbins or n x nbins(one to one)
% dist = n x 1

if size(P,2)~=size(Q,2)
    error('the number of columns in P and Q should be the same');
end

if sum(~isfinite(P(:))) + sum(~isfinite(Q(:)))
   error('the inputs contain non-finite values!') 
end

% normalizing the P and Q
if size(Q,1)==1
    Q = Q ./sum(Q);
    P = P ./repmat(sum(P,2),[1 size(P,2)]);
    temp =  P.*log(P./repmat(Q,[size(P,1) 1]));
    temp(isnan(temp))=0;% resolving the case when P(i)==0
    dist = sum(temp,2);
        
elseif size(Q,1)==size(P,1)
    
    Q = Q ./repmat(sum(Q,2),[1 size(Q,2)]);
    P = P ./repmat(sum(P,2),[1 size(P,2)]);
    temp =  P.*log(P./Q);
    temp(isnan(temp))=0; % resolving the case when P(i)==0
    dist = sum(temp,2);
end
end