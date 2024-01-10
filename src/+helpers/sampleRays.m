function dataOut = sampleRays(data,params)
% Copyright 2023 The MathWorks, Inc.
I = data{1};
I = im2single(I);
rgb = reshape(I,[],3);
tform = data{2};
od = helpers.computeRay(params.intrinsics,tform,params);
od = single(od);
od = reshape(od,[],9);
dataOut = [rgb,od]';
end