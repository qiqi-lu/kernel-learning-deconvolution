clear all
clc
specimen = 'Microtubules2';
dampar = 0;
num_iter = 6;
% 
% specimen = 'F-actin_Nonlinear';
% dampar = 0;
% num_iter = 8;

data_path = strcat(['F:\Datasets\BioSR\', specimen, '\']);
fig_path  = strcat(['E:\Project\2023 cytoSR\outputs\figures\',...
    lower(specimen), '\scale_1_gauss_9_poiss_1_ratio_1']);
data_path_txt = readlines([data_path, 'test.txt']);

id_data = [0,1,2,3,4,5,6,7,8,9,10];
% id_data = [0];
noise_level = 9;
scale_factor = 1;

%% Blind deconvolution
for i = id_data
    disp(i)
    img_raw = double(ReadTifStack(strcat(data_path,'raw_noise_',...
        num2str(noise_level), '\' ,data_path_txt(i+1))));
    img_gt  = double(ReadTifStack(strcat(data_path,'gt_sf_',...
        num2str(scale_factor), '\',data_path_txt(i+1))));
    psfi    = ones([5,5]);
    [img_deconv, psf] = deconvblind(img_raw, psfi, num_iter, dampar);
    tmp = strcat([fig_path, '\sample_', num2str(i), '\deconvblind\']);
    state = mkdir(tmp);
    save_to = strcat([tmp, 'deconv.tif']);
    WriteTifStack(img_deconv, save_to)
end

