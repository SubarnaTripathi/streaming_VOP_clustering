% Script that gives a demo of the Fast Video Segment algorithm
%
%    Copyright (C) 2013  Anestis Papazoglou
%
%    You can redistribute and/or modify this software for non-commercial use
%    under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%    For commercial use, contact the author for licensing options.
%
%    Contact: a.papazoglou@sms.ed.ac.uk

addpath( genpath( '.' ) )

foldername = fileparts( mfilename( 'fullpath' ) );

% The folder where the frames are stored in. Frames should be .jpg files
% and their names should be 8-digit sequential numbers (e.g. 00000001.jpg,
% 00000002.jpg etc)
options.infolder = fullfile( foldername, 'Data', 'inputs', 'youtube' );

% The folder where all the outputs will be stored.
options.outfolder = fullfile( foldername, 'Data', 'outputs', 'youtube_10_frames' );

% The optical flow method to be used. Valid names are:
%   broxPAMI2011:     CPU-based optical flow.
%   sundaramECCV2010: GPU-based optical flow. Requires CUDA 5.0
options.flowmethod = 'broxPAMI2011';

% The superpixel oversegmentation method to be used. Valid names are:
%   Turbopixels
%   SLIC
options.superpixels = 'SLIC';

% Create videos of the final segmentation and intermediate results?
% We recommend turning this option to false for your actual dataset, as
% rendering the video output is relatively computationally expensive.
options.visualise = true;

% Print status messages on screen
options.vocal = true;

% options.ranges:
%   A matlab array of length S+1, containing the number for the 
%   first frame of each shot (where S is the total count of shots
%   inside the options.infolder). 
%   The last element of the array should be equal to the total 
%   number of frames + 1.
options.ranges = [ 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]; %[1, 51 ];

% options.positiveRanges:
%		A matlab array containing the shots to be processed
options.positiveRanges = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ];

% If the frames are larger than options.maxedge in either height or width, they
% will be resized to fit a (maxedge x maxedge) window. This greatly decreases
% the optical flow computation cost, without (typically) degrading the
% segmentation accuracy too much. If resizing the frame is not desirable, set
% options.maxedge = inf
options.maxedge = 400;

% Use default params. For specific value info check inside the function
params = getDefaultParams();

% Create folder to save the segmentation
segmfolder = fullfile( options.outfolder, 'segmentations', 'VideoRapidSegment' );
if( ~exist( segmfolder, 'dir' ) ), mkdir( segmfolder ), end;

for( shot = options.positiveRanges )

    % Load optical flow (or compute if file is not found)
    data.flow = loadFlow( options, shot );
    if( isempty( data.flow ) )
        data.flow = computeOpticalFlow( options, shot );
    end
    
    % Load superpixels (or compute if not found)
    data.superpixels = loadSuperpixels( options, shot );
    if( isempty( data.superpixels ) )
        data.superpixels = computeSuperpixels( options, shot );
    end
    
    % Cache all frames in memory
    data.imgs = readAllFrames( options, shot );
    
    data.id = shot;
    segmentation = videoRapidSegment( options, params, data );
    
    % Save output
    filename = fullfile( segmfolder, sprintf( 'segmentationShot%i.mat', shot ) );
    save( filename, 'segmentation', '-v7.3' );
    
end

rmpath( genpath( '.' ) )
