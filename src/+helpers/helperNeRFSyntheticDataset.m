function [dsTrain, dsValid, params] = helperNeRFSyntheticDataset
% This helper function uses a labeled data set which comes from NeRF synthetic
% dataset
% Copyright 2023 The MathWorks, Inc.

datasetPath = fullfile("data","nerf_synthetic","lego");
if ~exist(datasetPath,"dir")
    error(string(datasetPath)+" is not found. "+newline+"Please download the NeRF synthetic dataset (nerf_synthetic.zip) " + ...
        string('from <a href="https://www.matthewtancik.com/nerf/">the website</a> and extract it to "data" folder.')) %#ok<STRQUOT>
end
[~,name,~] = fileparts(datasetPath);
trTrainJson = fileread(fullfile(datasetPath,"transforms_train.json"));
trValJson = fileread(fullfile(datasetPath,"transforms_val.json"));
trTrain = jsondecode(trTrainJson);
trVal = jsondecode(trValJson);
filePathTrain = arrayfun(@(x)string(x.file_path),trTrain.frames) + ".png";
filePathVal = arrayfun(@(x)string(x.file_path),trVal.frames) + ".png";
imdsTrain = imageDatastore(fullfile(datasetPath,filePathTrain));
I = read(imdsTrain);

params.modelName = name;
params.height = size(I,1);
params.width = size(I,2);
ncols = params.width;
nrows = params.height;
params.numCoarse = 64;
params.numFine = 128;
params.noise = 0;
params.sampleFromSingle = false;
params.roi = [-1,1,-1,1,-1,1]/2;
params.useViewDirs = true;
params.valStepPixels = 4;
params.valNumFrames = 1;

% Background color (should be white for DeepVoxel dataset, otherwise shoud
% be black)
params.bg = single([0; 0; 0]);

focal = params.width / tan(trTrain.camera_angle_x / 2) / 2;
fx_org = focal;
fy_org = focal;
cx_org = params.width / 2;
cy_org = params.height / 2;
k1 = 0;
k2 = 0;
p1 = 0;
p2 = 0;

fx = fx_org * params.width / ncols;
fy = fy_org * params.height / nrows;
cx = cx_org * params.width / ncols;
cy = cy_org * params.height / nrows;

intrinsicMatrix = [fx, 0, cx; 0, fy, cy; 0, 0, 1];
distortionCoefficients = [k1, k2, p1, p2, 0];
imageSize = [params.height, params.width];
params.intrinsics = cameraIntrinsicsFromOpenCV(intrinsicMatrix,distortionCoefficients,imageSize);

imdsTrain = imageDatastore(fullfile(datasetPath,filePathTrain),...
    "ReadFcn",@(x) undistortImage(imresize(imread(x),[params.height, params.width]),params.intrinsics));

imdsValid = imageDatastore(fullfile(datasetPath,filePathVal),...
    "ReadFcn",@(x) undistortImage(imresize(imread(x),[params.height, params.width]),params.intrinsics));

scale = 0.33;
camPosesTrain = cellfun(@(x)rigidtform3d([eul2rotm(rotm2eul([x(1:3,[2,1]),-x(1:3,3)]))*[0,-1,0;1,0,0;0,0,1],x(1:3,4)*scale; 0,0,0,1]),...
    {trTrain.frames.transform_matrix});
camPosesValid = cellfun(@(x)rigidtform3d([eul2rotm(rotm2eul([x(1:3,[2,1]),-x(1:3,3)]))*[0,-1,0;1,0,0;0,0,1],x(1:3,4)*scale; 0,0,0,1]),...
    {trVal.frames.transform_matrix});
dsPosesTrain = arrayDatastore(camPosesTrain');
dsPosesValid = arrayDatastore(camPosesValid');

dsTrain = combine(imdsTrain,dsPosesTrain);
dsValid = combine(imdsValid,dsPosesValid);
params.t_n = 0.05;
params.t_f = 2;

% Define a camera trajectory for novel view synthesis after training
rotCenter = [0,0,0];
N = 30*2;
numRot = 1;
angels = 2*pi*((1:N)'-1)*numRot/N;
zrange = 0.5*ones(N,1);
yrange = -1*ones(N,1);
xrange = 0*ones(N,1);
tr = [xrange,yrange,zrange];
pitch = pi - atan2(tr(:,3),tr(:,2));
poses = num2cell([tr, pitch, angels],2);
tforms = cellfun(@(x)rigidtform3d(trvec2tform(rotCenter)*...
    eul2tform([x(5),0,0])*...
    trvec2tform(-rotCenter)*...
    trvec2tform(rotCenter+x(1:3))*...
    eul2tform([0,0,-pi/2-x(4)])),poses);
params.testCamPoses = tforms;
end