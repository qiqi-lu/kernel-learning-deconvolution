%% set path and load some data
addpath('funcs')
% image = double(loadData('test_image.tif'));
% image = double(loadData("F:\Datasets\RLN\SimuMix3D_382\gt\1.tif"));
image = double(ReadTifStack("F:\Datasets\RLN\SimuMix3D_128\raw_psf_31_noise_0_sf_1_ratio_1\1.tif"));
image = permute(image, [3,1,2]);
% image = squeeze(max(image, [], 3));
image = squeeze(image(2,:,:));

pps = 5; % projected pixel size of 15nm
% typical parameters for resolution estimate
Nr = 50;
Ng = 10;
r = linspace(0,1,Nr);
GPU = 0;

%% apodize image edges with a cosine function over 20 pixels
image = apodImRect(image,20);

%% compute resolution

figID = 100;
if GPU 
    [kcMax,A0] = getDcorr(gpuArray(image),r,Ng,figID); gpuDevice(1);
else
    [kcMax,A0] = getDcorr(image,r,Ng,figID);
end

disp(['kcMax : ',num2str(kcMax,3),', A0 : ',num2str(A0,3)])
%% sectorial resolution

Na = 2; % number of sectors
figID = 1;
if GPU
    [kcMax,A0] = getDcorrSect(gpuArray(image),r,Ng,Na,figID); gpuDevice(1);
else
    [kcMax,A0] = getDcorrSect(image,r,Ng,Na,figID);
end

%% Local resolution map

tileSize = 100; % in pixels
tileOverlap = 0; % in pixels
figId = 103;

if GPU 
    [kcMap,A0Map] = getLocalDcorr(gpuArray(image),tileSize,tileOverlap,r,Ng,figID);gpuDevice(1);
else
    [kcMap,A0Map] = getLocalDcorr(image,tileSize,tileOverlap,r,Ng,figID);
end