function [dsTrain, dsValid, params] = helperReadDeepVoxelDataset
% This helper function loads a labeled data set which comes from DeepVoxels dataset[1]. Each image has the corresponding pose of the camera. Download the dataset and then extract it in a specified location below.
% https://www.vincentsitzmann.com/deepvoxels/
% [1] Sitzmann, V., Thies, J., Heide, F., Nießner, M., Wetzstein, G., Zollh¨ofer, M.: Deepvoxels: Learning persistent 3D feature embeddings. In: CVPR (2019)
% Copyright 2023 The MathWorks, Inc.

datasetPathTrain = fullfile("data","synthetic_scenes/train/greek");
datasetPathValidation = fullfile("data","synthetic_scenes/validation/greek");
if ~exist(datasetPathTrain,"dir")
    error(string(datasetPathTrain)+" is not found. "+newline+"Please download the Deep Voxel dataset (synthetic_senes.zip) " + ...
        string('from <a href="https://www.vincentsitzmann.com/deepvoxels/">the website</a> and extract it to "data" folder.')) %#ok<STRQUOT> 
end
[~,name,~] = fileparts(datasetPathTrain);

params.modelName = name;
params.height = 512;
params.width = 512;
params.numCoarse = 64;
params.numFine = 128;
params.noise = 0;
params.sampleFromSingle = true;
params.roi = [-1.5,1.5,-1.5,1.5,-1.5,1.5];
params.useViewDirs = true;
params.valStepPixels = 2;
params.valNumFrames = 1;

% Background color (should be white for DeepVoxel dataset, otherwise shoud
% be black)
params.bg = single([1; 1; 1]);

% Reduce the number of the validation images
numValidation = 1;

imdsTrain = imageDatastore(fullfile(datasetPathTrain,"rgb"),...
    "ReadFcn",@(x) imresize(imread(x),[params.height, params.width],"lanczos3"));
imds = imageDatastore(fullfile(datasetPathValidation,"rgb"));
imdsValidation = imageDatastore(imds.Files(1:numValidation),...
    "ReadFcn",@(x) imresize(imread(x),[params.height, params.width],"lanczos3"));

intrinsicsCell = readcell(fullfile(datasetPathTrain,"intrinsics.txt"));
f_org = intrinsicsCell{1,1};
cx_org = intrinsicsCell{1,2};
cy_org = intrinsicsCell{1,3};
org_x = intrinsicsCell{2,1};
org_y = intrinsicsCell{2,2};
org_z = intrinsicsCell{2,3};
near_plane = intrinsicsCell{1,1};
scale = intrinsicsCell{4,1};
nrows = intrinsicsCell{5,1};
ncols = intrinsicsCell{5,2};

imageSize = [params.height, params.width];
f = f_org * params.height / nrows;
cx = cx_org * params.width / ncols;
cy = cy_org * params.height / nrows;
params.intrinsics = cameraIntrinsics([f,f],[cx,cy],imageSize);

posedsTrain = fileDatastore(fullfile(datasetPathTrain,"pose"),...
    "FileExtensions",".txt",...
    "ReadFcn",@(x) helperReadPose(x));
poseds = fileDatastore(fullfile(datasetPathValidation,"pose"),...
    "FileExtensions",".txt",...
    "ReadFcn",@(x) helperReadPose(x));
posedsValidation = fileDatastore(poseds.Files(1:numValidation),...
    "FileExtensions",".txt",...
    "ReadFcn",@(x) helperReadPose(x));
dsTrain = combine(imdsTrain,posedsTrain);
dsValid = combine(imdsValidation,posedsValidation);


tforms = readall(posedsTrain);
distance = mean(cellfun(@(x)norm(x.Translation),tforms));
params.t_n = distance - 1;
params.t_f = distance + 1;

% Define a camera trajectory for novel view synthesis after training
rotCenter = [0,0,0];
N = 30*2;
numRot = 1;
angels = 2*pi*((1:N)'-1)*numRot/N;
zrange = 1*ones(N,1);
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

function tform = convertLefthandToRighthand(T)
    tform = T;
    tform(3,:) = -tform(3,:); % left-hand to right-hand
    rotm = tform2rotm(tform); % Extract the rotation matrix
    rotm_rigid = eul2rotm(rotm2eul(rotm)); % Non-rigid rotation matrix to rigid due to numerical errors
    tform(1:3,1:3) = rotm_rigid;
    tform = eul2tform([0,0,-pi/2]) * tform;
    tform = rigidtform3d(tform);
end

function tform = helperReadPose(poseFile)
    Tpose = reshape(table2array(readtable(poseFile,"Delimiter"," ")),4,4)';
    tform = convertLefthandToRighthand(Tpose);
end
end