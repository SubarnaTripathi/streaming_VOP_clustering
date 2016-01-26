function [D] = quantize_IoU_distance( IoU_mat, all_W )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    %% ease of access for sampling: maintain a list of all bbs corresponding to a IoU bins 
    %% this would be matrix of cell
    %% some entries may have multiple bounding boxes corresponding to a particular IoU bin wrt a particular bb
    %% some can even be blank

    bin_num = 10; % uniform quantization of the range [0-1]
    bb_num = size(IoU_mat, 1);

    D = cell(bb_num, bin_num);    

    for bb_index = 1:bb_num        
        for i = 1:bin_num 
            eval(sprintf('clear l%d',i));
        end        
        %clear l1 l2 l3 l4 l5 l6 l7 l8 l9 l10;
        % initialize for every bb index
        %%
        for aux_bb_index = 1:bb_num
            IoU_val = IoU_mat(bb_index, aux_bb_index);
            bin_index = min(bin_num, floor(IoU_val*bin_num + 0.5));  %% based on size : bin_num = 10
            bin_index = max(bin_index, 1);  %% based on size : bin_num = 10
            
            temp = D(bb_index,bin_index);
            t = temp{1};
            
            if( isempty(t) )
                curr = 0;
            else
                fields = fieldnames(t); %(D(bb_index, bin_index));
                curr = numel(fields);
            end
            loc_var = sprintf('w%d',curr+1);
            %bb_loc = all_W(aux_bb_index,:); %% subarna
            bb_loc = aux_bb_index; %% store only bb index number and not the bb corners
 
            %% short version
            list_name = sprintf('l%d',bin_index);           
            eval(sprintf('%s.(loc_var) = %d;',list_name, bb_loc));            
            eval(sprintf('D(bb_index,bin_index) = {%s};', list_name));
            %%
            
%%            
%             %% elaborate version : debug
%             switch(bin_index)
%                 case 1
%                     l1.(loc_var) = bb_loc; 
%                     D(bb_index, 1) = {l1}; 
%                 case 2
%                     l2.(loc_var) = bb_loc;  
%                     D(bb_index, 2) = {l2}; 
%                 case 3
%                     l3.(loc_var) = bb_loc;
%                     D(bb_index, 3) = {l3};  
%                 case 4
%                     l4.(loc_var) = bb_loc;  
%                     D(bb_index, 4) = {l4}; 
%                 case 5
%                     l5.(loc_var) = bb_loc;
%                     D(bb_index, 5) = {l5}; 
%                 case 6
%                     l6.(loc_var) = bb_loc;
%                     D(bb_index, 6) = {l6}; 
%                 case 7
%                     l7.(loc_var) = bb_loc; 
%                     D(bb_index, 7) = {l7};
%                 case 8
%                     l8.(loc_var) = bb_loc; 
%                     D(bb_index, 8) = {l8}; 
%                 case 9
%                     l9.(loc_var) = bb_loc;
%                     D(bb_index, 9) = {l9}; 
%                 case 10
%                     l10.(loc_var) = bb_loc;  
%                     D(bb_index, 10) = {l10}; 
%             end 
%%
        end
    end

end

