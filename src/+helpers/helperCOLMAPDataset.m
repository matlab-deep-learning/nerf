function [dsTrain, dsValid, params] = helperCOLMAPDataset
% This helper funtion loads dataset created using COLMAP
% Copyright 2023 The MathWorks, Inc.

datasetPathTrain = fullfile("data","fox");
if ~exist(datasetPathTrain,"dir")
    error(string(datasetPathTrain)+" is not found. "+newline+"Please prepare the COLMAP dataset " + ...
        string('from <a href="https://github.com/NVlabs/instant-ngp/tree/master/data/nerf/fox">the website</a> and extract it to "data" folder.')) %#ok<STRQUOT>
end
[~,name,~] = fileparts(datasetPathTrain);
text = fileread(fullfile(datasetPathTrain,"transforms.json"));
value = jsondecode(text);
imds = imageDatastore(fullfile(datasetPathTrain,string(cat(1,value.frames.file_path))));
I = read(imds);

params.modelName = name;
params.height = size(I,1);
params.width = size(I,2);
ncols = value.w;
nrows = value.h;
params.numCoarse = 64;
params.numFine = 128;
params.noise = 0;
params.sampleFromSingle = false;
params.roi = [-1,1,-1,1,-1,1]*4;
params.useViewDirs = true;
params.valStepPixels = 6;
params.valNumFrames = 1;

% Background color (should be white for DeepVoxel dataset, otherwise shoud
% be black)
params.bg = single([0; 0; 0]);

fx_org = value.fl_x;
fy_org = value.fl_y;
cx_org = value.cx;
cy_org = value.cy;
k1 = value.k1;
k2 = value.k2;
p1 = value.p1;
p2 = value.p2;

imageSize = [params.height, params.width];
fx = fx_org * params.width / ncols;
fy = fy_org * params.height / nrows;
cx = cx_org * params.width / ncols;
cy = cy_org * params.height / nrows;

intrinsicMatrix = [fx, 0, cx; 0, fy, cy; 0, 0, 1];
distortionCoefficients = [k1, k2, p1, p2, 0];
imageSize = [params.height, params.width];
params.intrinsics = cameraIntrinsicsFromOpenCV(intrinsicMatrix,distortionCoefficients,imageSize);

imdsTrain = imageDatastore(fullfile(datasetPathTrain,string(cat(1,value.frames.file_path))),...
    "ReadFcn",@(x) undistortImage(imresize(imread(x),[params.height, params.width]),params.intrinsics));

scale = 0.33;
camPoses = cellfun(@(x)rigidtform3d([eul2rotm(rotm2eul([x(1:3,[2,1]),-x(1:3,3)]))*[0,-1,0;1,0,0;0,0,1],x(1:3,4)*scale; 0,0,0,1]),...
    {value.frames.transform_matrix});
poseds = arrayDatastore(camPoses');

dsTrain = combine(imdsTrain,poseds);
dsValid = dsTrain;
params.t_n = 0;
params.t_f = max(params.roi(2:2:end)-params.roi(1:2:end))*sqrt(3);

% Define a camera trajectory for novel view synthesis after training
locationCenter = [1.2 -2.082 -0.3231];
orientation = eul2rotm([0.4637    0.0625   -1.4985]);
N = 30*2;
numRot = 1;
angles = 2*pi*((1:N)'-1)*numRot/N;
location = locationCenter + [0.2*cos(angles), zeros(size(angles)), 0.2*sin(angles)];
locationCell = num2cell(location,2);
tforms = cellfun(@(x)rigidtform3d(orientation,x),locationCell);
params.testCamPoses = tforms;
end