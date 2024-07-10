%% Load data
clear all
clc

% data_path = 'F:\Datasets\BioSR\Microtubules2\';
data_path = 'F:\Datasets\BioSR\F-actin_Nonlinear\';

data_path_txt = readlines([data_path, 'test.txt']);
id_data = 6;

noise_level  = 9;
scale_factor = 1;
img_raw = double(ReadTifStack(strcat(data_path, 'raw_noise_',...
    num2str(noise_level), '\', data_path_txt(id_data))));
img_gt  = double(ReadTifStack(strcat(data_path, 'gt_sf_',...
    num2str(scale_factor), '\', data_path_txt(id_data))));
disp(['Load data from: ', data_path])

%% Blind deconvolution
psfi   = ones([5,5]);
iter   = 3;
dampar = 0;

disp(['Niter :', num2str(iter), '|', ' Dampar :', num2str(dampar)])
[img_deconv, psf] = deconvblind(img_raw, psfi, iter, dampar);

disp(['NCC|MSE (raw) : ', num2str(NCC(img_raw, img_gt)), '|', ...
    num2str(immse(img_raw, img_gt))])
disp(['NCC|MSE (deconv) : ', num2str(NCC(img_deconv, img_gt)), '|', ...
    num2str(immse(img_deconv, img_gt))])
disp('end')

%% show result
img_max = max(img_gt(:)) * 0.8;
figure()
subplot(2,2,1)
imshow(squeeze(img_raw),[0,img_max]),colorbar
title('RAW')
subplot(2,2,2)
imshow(squeeze(img_deconv),[0,img_max]),colorbar
title('Deconv')
subplot(2,2,3)
imshow(squeeze(img_gt),[0,img_max]),colorbar
title('GT')
subplot(2,2,4)
imshow(squeeze(psf),[]),colorbar

%% Blind deconvolution
psfi   = ones([5,5]);
% psfi   = ones([31,31]);
N_iter = 20;
N_dampar = 2;

metrics1 = zeros([N_iter,N_dampar]);
metrics2 = metrics1;

disp(['NCC of the raw image : ', num2str(NCC(img_raw, img_gt))])
disp(['MSE of the raw image : ', num2str(immse(img_raw, img_gt))])

for i=1:N_iter
    disp(i)
    for j=1:N_dampar
        [img_deconv, psf] = deconvblind(img_raw, psfi, i, (j-1)*1);
        tmp1 = immse(img_deconv, img_gt);
        tmp2 = NCC(img_deconv, img_gt);
        % disp(tmp)
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
xlabel('iter')
ylabel('MSE')
subplot(1,2,2)
plot(metrics2)
hold on
legend('1','2','3','4','5')
xlabel('iter')
ylabel('NCC')