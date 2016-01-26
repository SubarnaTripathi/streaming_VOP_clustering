%% function [F,F_unary] = extractF(f_maps,ii,jj)
%
% INPUTS
%  f_maps - NxMxF array of F feature maps for an NxM image
%  ii     - indices of samples for feature A
%  jj     - indices of samples for feature B
%  
% OUTPUS
% F        - each row is a feature pair {A,B}
%              A and B each have equal dimensionality
% F_unary  - each row is a feature A or B
%
% -------------------------------------------------------------------------
% Crisp Boundaries Toolbox
% Phillip Isola, 2014 [phillpi@mit.edu]
% Please email me if you find bugs, or have suggestions or questions
% -------------------------------------------------------------------------

function [F,F_unary] = extractF_ST(f_maps,ii,jj)

    %Npixels = numel(f_maps(:,:,1));
    Npixels = size(f_maps,1);
    
    %% extract features
    %F = zeros(size(ii,1),size(f_maps,3)*2);
    F = zeros(size(ii,1),size(f_maps,2)*2);
    for c=1:size(f_maps,2) %% subarna: 3
        %tmp = permute(f_maps(:,:,c),[2 1 3]);
        tmp = permute(f_maps(:,c), [2 1]);
        F(:,c) = tmp(ii);
        F(:,c+size(f_maps,2)) = tmp(jj);
    end
    %keyboard
    
    kk = cat(1,ii,jj); % only need to evaluate on indices that ii or jj refer to (see usage in evalPMI)
    F_unary = nan(Npixels,size(f_maps,2)); %% subarna: 3
    for c=1:size(f_maps,2) %% %% subarna: 3
        %tmp1 = permute(f_maps(:,:,c),[2 1 3]);
        tmp1 = permute(f_maps(:,c),[2 1]);
        tmp2 = F_unary(:,c);
        tmp2(kk) = tmp1(kk);
        F_unary(:,c) = tmp2;
    end
    
    %keyboard
    
    %% order A and B so that we only have to model half the space (assumes
    % symmetry: p(A,B) = p(B,A))
    F = orderAB(F);
end