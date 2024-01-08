function data = computeRay(intrinsics, tform, params)
% Copyright 2023 The MathWorks, Inc.
imageSize = single(intrinsics.ImageSize);
[u,v] = meshgrid(0:imageSize(2)-1,0:imageSize(1)-1);
xy = (cat(3,u,v) - permute(single(intrinsics.PrincipalPoint),[1,3,2])) ./ permute(single(intrinsics.FocalLength),[1,3,2]);
xyz = cat(3,xy,ones([size(xy,1:2),1]));
d = reshape(tform.transformPointsForward(reshape(xyz,[],3)),size(xyz));
o = permute(tform.Translation,[1,3,2]);
d = d - o;
d_norm = sqrt(sum(d.^2,3));
t_n = params.t_n * ones(size(d_norm));
t_f = params.t_f * ones(size(d_norm));
data = cat(3,repmat(o,imageSize(1:2)),d,d_norm,t_n,t_f);
end