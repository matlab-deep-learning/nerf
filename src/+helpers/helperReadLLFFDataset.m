function [dsTrain, dsValid, params] = helperReadLLFFDataset
% This example uses a labeled data set which comes from LLFF dataset[1]. Each image has the corresponding pose of the camera. Download the dataset and then extract it in a specified location below.
% [1] Mildenhall, Ben, et al. "Local light field fusion: Practical view synthesis with prescriptive sampling guidelines." ACM Transactions on Graphics (TOG) 38.4 (2019): 1-14.
% Copyright 2023 The MathWorks, Inc.

datasetPath = fullfile("data","nerf_llff_data/fern");
if ~exist(datasetPath,"dir")
    error(string(datasetPath)+" is not found. "+newline+"Please download the LLFF dataset (nerf_llff_data.zip) " + ...
        string('from <a href="https://www.matthewtancik.com/nerf/">the website</a> and extract it to "data" folder.')) %#ok<STRQUOT>
end
[~,name,~] = fileparts(datasetPath);

factor = 8;
params.modelName = name;
params.height = 378;
params.width = 504;
params.numCoarse = 64;
params.numFine = 64;
params.noise = 0;
params.roi = [-1,1,-1,1,-1,1]*4;
params.useViewDirs = true;

% Background color (should be white for DeepVoxel dataset, otherwise shoud
% be black)
params.bg = single([0; 0; 0]);

% Setup an NPY file reader
filename = 'npy-matlab.zip';
url = 'https://github.com/kwikteam/npy-matlab/archive/master.zip';
if ~exist('npy-matlab-master','dir')
    outfilename = websave(filename,url);
    unzip(filename)
end
addpath('npy-matlab-master/npy-matlab')

poses_hwf_nf = readNPY(fullfile(datasetPath,"poses_bounds.npy"));
poses_hwf = permute(reshape(poses_hwf_nf(:,1:15),[],5,3),[3,2,1]);
poses_hwf(end,end,:) = poses_hwf(end,end,:) / factor;
poses = [poses_hwf(:,1:4,:); repmat([0,0,0,1],[1,1,size(poses_hwf,3)])];
scale = 1/(min(poses_hwf_nf(:,end-1)) * 0.75);
poses(1:3,4,:) = poses(1:3,4,:) * scale;
poses = mat2cell(poses,4,4,ones(size(poses,3),1));
poses = cellfun(@(x) [x(:,2),x(:,1),-x(:,3), x(:,4)], poses, "UniformOutput",false);
poses = squeeze(cellfun(@(x)rigidtform3d(x),poses));
posedsTrain = arrayDatastore(poses);
params.valStepPixels = 2;
params.valNumFrames = 1;

f_org = poses_hwf(3,5,1);
cx_org = poses_hwf(2,5,1) / 2;
cy_org = poses_hwf(1,5,1) / 2;
nrows = poses_hwf(1,5,1);
ncols = poses_hwf(2,5,1);

params.t_n = 0;
params.t_f = 1;

imageSize = [params.height, params.width];
f = f_org * params.height / nrows;
cx = cx_org * params.width / ncols;
cy = cy_org * params.height / nrows;
params.intrinsics = cameraIntrinsics([f,f],[cx,cy],imageSize);
imdsTrain = imageDatastore(fullfile(datasetPath,"images"),...
    "ReadFcn",@(x) imresize(imread(x),[params.height, params.width],"lanczos3"));

dsTrain = combine(imdsTrain,posedsTrain);
dsValid = dsTrain;

% Define a camera trajectory for novel view synthesis after training
locationCenter = [0,0,0];
orientation = eul2rotm([0,0,0]);
N = 30*2;
numRot = 1;
angles = 2*pi*((1:N)'-1)*numRot/N;
location = locationCenter + [0.1*cos(angles), 0.1*sin(angles), zeros(size(angles))];
locationCell = num2cell(location,2);
tforms = cellfun(@(x)rigidtform3d(orientation,x),locationCell);
params.testCamPoses = tforms;

warning("Forward facing scenes like LLFF need to be projected to NDC space for better convergence.");

end