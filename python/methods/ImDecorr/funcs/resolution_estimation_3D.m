function [kcMax_xy, kcMax_z] = resolution_estimation_3D(image, Nr, Ng, projection, figID)
% (custom function)
% The lateral and axial resolution of value calculated from the
% maximum-intensity projections. (Li, et al, 2022.)

% image, 3D image with dimeisona [Nz, Ny, Nx]
% pps, projected pixel size of 15nm
% typical parameters for resolution estimate
% Nr = 50;
% Ng = 10;

% -------------------------------------------------------------------------
r = linspace(0,1,Nr);
[Nz, ~, Nx] = size(image);

if projection == 1
    disp('Estimate the resolution of the projection of slices.')
    % compute laterial resolution
    image_xy = squeeze(max(image, [], 1));
    % apodize image edges with a cosine function over 20 pixels
    image_xy = apodImRect(image_xy,20);
    [kcMax_xy, ~] = getDcorr(image_xy,r,Ng,figID);

    % compute axial resolution
    image_zy = squeeze(max(image, [], 3));
    image_zy = apodImRect(image_zy,20);
    Na = 2; % number of sectors
    [kcMax, ~] = getDcorrSect(image_zy,r,Ng,Na);
    kcMax_z = kcMax(2);
end

if projection == 0
    disp('Estimate the resolution of each slice.')
    disp('xy plane ...')

    if Nz>10
        start = floor(Nz/2)-5;
        stop  = floor(Nz/2)+5;
        kcMax_xy  = ones(11,1);
    else
        start = 1;
        stop  = Nz;
        kcMax_xy  = ones(Nz,1);
    end

    for i = start:stop
        image_xy = squeeze(image(i,:,:));
        image_xy = apodImRect(image_xy,20);
        [kcMax, ~] = getDcorr(image_xy,r,Ng);
        kcMax_xy(i-start+1) = kcMax;
    end

    disp('zy plane ...')
    if Nx>10
        start = floor(Nx/2)-5;
        stop  = floor(Nx/2)+5;
        kcMax_z = ones(11,1);
    else
        start = 1;
        stop  = Nx;
        kcMax_z = ones(Nx,1);
    end

    for i = start:stop
        image_zy = squeeze(image(:,:,i));
        image_zy = apodImRect(image_zy,20);
        Na = 2;
        [kcMax, ~] = getDcorrSect(image_zy,r,Ng,Na);
        kcMax_z(i-start+1) = kcMax(2);
    end
end
