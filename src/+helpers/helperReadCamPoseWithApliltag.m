function [dsTrain, dsValid, params] = helperReadCamPoseWithApliltag
% This helper function laods dataset created with AprilTag
% Copyright 2023 The MathWorks, Inc.

params.modelName = "x_pipe";
params.width = 1920;
params.height = 1080;
params.roi = [-1,1,-1,1,-1,1];
%longest = max(params.roi(2:2:end)-params.roi(1:2:end));
params.t_n = 0.01;
params.t_f = 3;
params.numCoarse = 64;
params.numFine = 128;
params.noise = 0;
params.useViewDirs = true;
params.valStepPixels = 8;
params.valNumFrames = 1;

% Background color (should be white for DeepVoxel dataset, otherwise shoud
% be black)
params.bg = single([0; 0; 0]);

% Retrieve the original image size
imageDir = fullfile("data",params.modelName,"image");
if ~exist(imageDir,"dir")
    error("'"+string(imageDir)+"' folder is not found. "+...
        "Run <a href=""matlab: edit NeRF_CameraPoseEstimationUsingAprilTag.mlx"">NeRF_CameraPoseEstimationUsingAprilTag.mlx</a> first to generate dataset.");
end
imds = imageDatastore(imageDir);
I = readimage(imds,1);
nrows = size(I,1);
ncols = size(I,2);

% Retrieve camera poses and rescale to meter
d = load(fullfile("data",params.modelName,"camPoses.mat"));
camPoses = d.camPoses;
camPoses = arrayfun(@(x) rigidtform3d(eul2tform([0,0,pi])*x.A),camPoses); % Align Z-axis to be upper side

% Rescale the intrinsics parameters
d = load(fullfile("data",params.modelName,"intrinsics.mat"));
imageSize = [params.height, params.width];
f = d.intrinsics.FocalLength .* [params.width / ncols, params.height / nrows];
c = d.intrinsics.PrincipalPoint .* [params.width / ncols, params.height / nrows];
params.intrinsics =  cameraIntrinsics(f,c,imageSize,"RadialDistortion",d.intrinsics.RadialDistortion);

% Create a combined datastore
% TODO: apply undistortImage (Naive undistortion generates invalid black
% edges. To avoid this, masks for loss need to be introduced.)
imds = imageDatastore(imds.Files,...
    "ReadFcn",@(x) imresize(imread(x),[params.height, params.width],"lanczos3"));
poseds = arrayDatastore(camPoses);
ds = combine(imds,poseds);

rng('default');
indicesRand = randperm(ds.numpartitions);
indexTrain = 1:(ds.numpartitions-1);
indexValid = ds.numpartitions;
dsTrain = subset(ds,indicesRand(indexTrain));
dsValid = subset(ds,indicesRand(indexValid));

% Define a camera trajectory for novel view synthesis after training
rotCenter = [0,0,0.1];
N = 30;
numRot = 2;
angels = 2*pi*((1:(N+1))'-1)*numRot/(N+1);
angels = angels(1:end-1);
zrange = linspace(0.15,-0.05,N)';
yrange = -0.12*ones(N,1);
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