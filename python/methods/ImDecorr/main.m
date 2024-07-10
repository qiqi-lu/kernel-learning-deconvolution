%% set path and load data
clear all
addpath('funcs')

%% set parameters
pps = 5; % projected pixel size of 15nm

% typical parameters for resolution estimate
Nr = 50;
Ng = 10;

%% resolution estimation (single)
image = double(ReadTifStack("F:\Datasets\RLN\SimuMix3D_128\raw_psf_31_noise_0.5_sf_1_ratio_0.1\1.tif"));
image = permute(image, [3,1,2]);
[kcMax_xy, kcMax_z] = resolution_estimation_3D(image, Nr, Ng, 0, 0);

disp(['Lateral resolution: ', num2str(mean(kcMax_xy(10:end-10))), ', Axial resolution:',  num2str(mean(kcMax_z(10:end-10)))])

%% resolution estimation (simulation data)
% data_path = 'E:\Project\2023 cytoSR\outputs\figures\simumix3d_128\scale_1_noise_0_ratio_1\';
% data_path = "E:\Project\2023 cytoSR\outputs\figures\simumix3d_128\scale_1_noise_0.5_ratio_1\";
% data_path = "E:\Project\2023 cytoSR\outputs\figures\simumix3d_128\scale_1_noise_0.5_ratio_0.3\";
% data_path = "E:\Project\2023 cytoSR\outputs\figures\simumix3d_128\scale_1_noise_0.5_ratio_0.1\";

% data_path = 'E:\Project\2023 cytoSR\outputs\figures\simubeads3d_128\scale_1_noise_0_ratio_1\';
% data_path = "E:\Project\2023 cytoSR\outputs\figures\simubeads3d_128\scale_1_noise_0.5_ratio_1\";
% data_path = "E:\Project\2023 cytoSR\outputs\figures\simubeads3d_128\scale_1_noise_0.5_ratio_0.3\";
data_path = "E:\Project\2023 cytoSR\outputs\figures\simubeads3d_128\scale_1_noise_0.5_ratio_0.1\";

meths = ["traditional"; "gaussian"; "butterworth"; "wiener_butterworth"; "kernelnet"];

%% resolution estimation (Real data)
data_path = 'E:\Project\2023 cytoSR\outputs\figures\microtubule\scale_1_noise_0_ratio_1';
% data_path = "E:\Project\2023 cytoSR\outputs\figures\nuclear_pore_complex\scale_1_noise_0_ratio_1";

meths = ["deconvblind"; "kernelnet"];

%% run
disp('Load data from: ')
disp(data_path)

id_sample = [0; 1; 2; 3; 4; 5];

Nmeth = size(meths, 1);
Nsample = size(id_sample,1);

for ids = 1:Nsample
    kcMax_xy_all = [];
    kcMax_z_all = [];
    disp(['sample: ', num2str(id_sample(ids))])
    
    % raw
    disp('RAW')
    img_path = strcat(data_path,'sample_',num2str(id_sample(ids)),'\',meths(Nmeth),'\x.tif');
    image = double(ReadTifStack(img_path));
    image = permute(image, [3,1,2]);
    [kcMax_xy, kcMax_z] = resolution_estimation_3D(image, Nr, Ng, 0, 0);
    kcMax_xy_all = cat(2, kcMax_xy_all, kcMax_xy);
    kcMax_z_all  = cat(2, kcMax_z_all, kcMax_z);
    
    % conventional methods
    for i = 1: (Nmeth-1)
        disp(meths(i))
        img_path = strcat(data_path,'sample_',num2str(id_sample(ids)),'\',meths(i),'\deconv.tif');
        image = double(ReadTifStack(img_path));
        image = permute(image, [3,1,2]);
        [kcMax_xy, kcMax_z] = resolution_estimation_3D(image, Nr, Ng, 0, 0);
        kcMax_xy_all = cat(2, kcMax_xy_all, kcMax_xy);
        kcMax_z_all  = cat(2, kcMax_z_all, kcMax_z);
    end
    
    % kernelnet
    disp(meths(Nmeth))
    img_path = strcat(data_path,'sample_',num2str(id_sample(ids)),'\',meths(Nmeth),'\y_pred_all.tif');
    image = double(tiffreadVolume(img_path));
    image = squeeze(image(:,:,size(image,3),:));
    image = permute(image, [3,1,2]);
    [kcMax_xy, kcMax_z] = resolution_estimation_3D(image, Nr, Ng, 0, 0);
    kcMax_xy_all = cat(2, kcMax_xy_all, kcMax_xy);
    kcMax_z_all  = cat(2, kcMax_z_all, kcMax_z);
    
    save_to = strcat(data_path,'sample_',num2str(id_sample(ids)));
    disp('Save to')
    disp(save_to)
    save(strcat(save_to, '\res_xy.mat'), "kcMax_xy_all")
    save(strcat(save_to, '\res_z.mat'), "kcMax_z_all")
end

