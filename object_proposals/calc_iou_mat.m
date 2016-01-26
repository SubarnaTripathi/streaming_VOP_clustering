%%
% I = input image (color)
% dt1 = detected bounding boxes from method 1
% dt2 = detected bb from method 2
% affinity = IoU score
%%
function [ IoU_mat ] = calc_iou_mat( I, dt1, dt2 )
%function [ IoU_mat ] = calc_iou_mat()

%bb_size1 = size(dt1,1); bb_size2 = size(dt2,1);
%affinity = zeros(bb_size1+bb_size2, bb_size1+bb_size2);
% dt=[0 0 10 10]; gt=[0 0 20 20];
% oa0 = bbGt('compOas',dt,gt,0);
%affinity(1,1) = oa0;

%bb = [dt1; dt2];
bb = [dt1; dt2];
all_bb_num = size(bb,1);

% bb1 = bb(1,:); bb2 = bb(2,:);
% affinity = bbGt('compOas',bb1,bb2,0); %% debug

bb1 = bb; bb2 = bb;
IoU_mat = bbGt('compOas',bb1,bb2,zeros(all_bb_num,1));

yellow = uint8([255 255 0]); % [R G B]; class of yellow detected bb from method 1
blue = uint8([0 0 255]); % [R G B]; class of blue detected bb from method 2

shapeInserter1 = vision.ShapeInserter('Shape','Rectangles','BorderColor','Custom','CustomBorderColor',yellow);
shapeInserter2 = vision.ShapeInserter('Shape','Rectangles','BorderColor','Custom','CustomBorderColor',blue);

%shapeInserter = vision.ShapeInserter;
% I = imread('cameraman.tif');
% RGB = repmat(I,[1,1,3]); % convert I to an RGB image

%rectangle = int32([0 0 10 10; 0 0 20 20]);
% rectangle = int32([dt; gt]);
% J = step(shapeInserter, I, rectangle);

rectangle1 = int32(dt1);
rectangle2 = int32(dt2);
J = step(shapeInserter1, I, rectangle1);
J = step(shapeInserter2, J, rectangle2);

%figure, imshow(J)

end

