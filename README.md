# streaming_VOP_clustering

This code generates moving and static object proposals (VOP) in generic video and clusters those proposals in a streaming way described in "Detecting Temporally Consistent Objects in Videos through Object Class Label Propagation"
http://arxiv.org/pdf/1601.05447.pdf.

Streaming clustering of VOP can enable object class label propagation for temporally consistent object detection in videos.
The code is tested on Ubuntu 14.04
The MATLAB script vid_obj.m is the starting file. All the dependencies are already available in the package. The demo runs for videos downloaded from Youtube. The input and output folder are there in "Data".

Different configuration options can be turned ON or OFF in the options structure in vid_obj: e.g. 

options.segmentation = 1/0 means Enable or Disbale segmentation results generation as a by-product of clustering 

options.debug = 1/0 means Enable or Disbale debugging with all intermediate results

options.eb_thresh sets the threshold needed for edgeBoxes score (default is 0.25) 

options.auto_cluster = 1/0 means Enable or Disbale automatic number of cluster usage 

options.volume_cluster = 0/1 means clustering on single frame and entire sub-sequence respectively 

options.volume_frames sets the number of frames in a sub-sequence for streaming volume clustering

###################################################################
This code uses a affinity based clustering part from "Crisp Boundary Detection Using Pointwise Mutual Information" Phillip Isola, Daniel Zoran, Dilip Krishnan, and Edward H. Adelson ECCV, 2014 
For further details refer to :https://github.com/phillipi/crisp-boundaries
###################################################################

