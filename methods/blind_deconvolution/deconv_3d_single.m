%% Load data
clear all
clc
% data_path = 'F:\Datasets\RCAN3D\Confocal_2_STED\Microtubule\';
% data_path = 'F:\Datasets\RCAN3D\Confocal_2_STED\Nuclear_Pore_complex\';

data_path = 'F:\Datasets\RCAN3D\Confocal_2_STED\Microtubule2\';
% data_path = 'F:\Datasets\RCAN3D\Confocal_2_STED\Nuclear_Pore_complex2\';

data_path_txt = readlines([data_path, 'test_1024x1024.txt']);
id_data = 8;

img_raw = double(ReadTifStack(strcat(data_path, 'raw_1024x1024\',...
    data_path_txt(id_data))));
img_gt  = double(ReadTifStack(strcat(data_path, 'gt_1024x1024\',...
    data_path_txt(id_data))));

disp(['Load data from: ', data_path])

%% Blind deconvolution
psfi = ones([11,11,5]);
iter = 3;
dampar = 0;
disp(['Niter :', num2str(iter), '|', ' num of Dampar :', num2str(dampar)])

[img_deconv, psf] = deconvblind(img_raw, psfi, iter, dampar);

disp(['NCC|MSE (raw) : ', num2str(NCC(img_raw, img_gt)), '|', ...
    num2str(immse(img_raw, img_gt))])
disp(['NCC|MSE (deconv) : ', num2str(NCC(img_deconv, img_gt)), '|', ...
    num2str(immse(img_deconv, img_gt))])
disp('end')

%% show result
img_max = max(img_gt(:)) * 0.8;
id_slice = 3;
figure()
subplot(2,2,1)
imshow(squeeze(img_raw(:,:,id_slice)),[0,img_max]),colorbar
title('RAW')
subplot(2,2,2)
imshow(squeeze(img_deconv(:,:,id_slice)),[0,img_max]),colorbar
title('DeconvBlind')
subplot(2,2,3)
imshow(squeeze(img_gt(:,:,id_slice)),[0,img_max]),colorbar
title('GT')
subplot(2,2,4)
imshow(squeeze(psf(:,:,2)),[]),colorbar

%% Blind deconvolution
psfi   = ones([11,11,5]);

N_iter = 10;
N_dampar = 1;

metrics1 = zeros([N_iter,N_dampar]);
metrics2 = zeros([N_iter,N_dampar]);

disp(['NCC of the raw image : ', num2str(NCC(img_raw, img_gt))])
disp(['MSE of the raw image : ', num2str(immse(img_raw, img_gt))])

for i=1:N_iter
    for j=1:N_dampar
        disp([i, j])
        [img_deconv, psf] = deconvblind(img_raw, psfi, i, (j-1)*0.1);
        tmp1 = immse(img_deconv, img_gt);
        tmp2 = NCC(img_deconv, img_gt);
        disp([num2str(tmp1), '|', num2str(tmp2)])
        metrics1(i,j) = tmp1;
        metrics2(i,j) = tmp2;
    end
end
disp('end')

%% plot mse
figure(1)

subplot(1,2,1)
plot(metrics1)
hold on
legend('1','2','3','4','5')
xlabel('num of iteration')
ylabel('MSE')

subplot(1,2,2)
plot(metrics2)
title('NCC')
hold on
legend('1','2','3','4','5')
xlabel('num of iteration')
ylabel('NCC')