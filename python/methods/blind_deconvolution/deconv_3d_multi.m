clear all
clc
% specimen = 'Microtubule';
specimen = 'Microtubule2';
% dampar = 0;
% num_iter = 3;

% specimen = 'Nuclear_Pore_complex';
% specimen = 'Nuclear_Pore_complex2';
dampar = 0;
num_iter = 3;

data_path = strcat(['F:\Datasets\RCAN3D\Confocal_2_STED\', specimen, '\']);
fig_path  = strcat(['E:\Project\2023 cytoSR\outputs\figures\',...
    lower(specimen), '\scale_1_gauss_0_poiss_0_ratio_1']);
data_path_txt = readlines([data_path, 'test_1024x1024.txt']);

id_data = [0,1,2,3,4,5, 6];
% id_data = [6,7,8,9,10,11,12,13,14,15,16];
% id_data = [0];

%% Blind deconvolution
for i = id_data
    disp(i)
    img_raw = double(ReadTifStack(strcat(data_path,'raw_1024x1024\',...
        data_path_txt(i+1))));
    img_gt  = double(ReadTifStack(strcat(data_path,'gt_1024x1024\',...
        data_path_txt(i+1))));
    psfi    = ones([11,11,5]);
    [img_deconv, psf] = deconvblind(img_raw, psfi, num_iter, dampar);
    tmp = strcat([fig_path, '\sample_', num2str(i), '\deconvblind\']);
    state = mkdir(tmp);
    save_to = strcat([tmp, 'deconv.tif']);
    WriteTifStack(img_deconv, save_to)
end

