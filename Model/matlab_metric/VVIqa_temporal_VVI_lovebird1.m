clc;
clear;
close all

p = pwd;
addpath(fullfile(p, '/code_MPPSNR')) % MPPSNR algorithm
addpath(fullfile(p, '/code_MWPSNR')) % MWPSNR algorithm
addpath(fullfile(p, '/iwssim_iwpsnr')) % iwssim algorithm
addpath(fullfile(p, '/msssim')) % iwssim algorithm
addpath(fullfile(p, '/matlabPyrTools-master')) %
addpath(fullfile(p, '/matlabPyrTools-master/MEX')) %
saveFileName = 'outputs/lovebird1_allvideos.txt';
orgVideoPath = 'E:\Dataset\MVDsequences_Imgs\H265\lovebird1\gt_gray\';

testvideo = {
    '1';
    '2';
    '3';
    '4';
    '5';
    };

orgFiles=dir([orgVideoPath '*.png']);  % xu zhu yi
Numframe = length(orgFiles);
Numframe1 = Numframe;
MP_PSNR_full = zeros(Numframe1,1);
MP_PSNR_reduc = zeros(Numframe1,1);


ave_MP_PSNR = zeros(length(testvideo),1);
ave_MP_PSNR_reduc = zeros(length(testvideo),1);


IW_SSIM = zeros(Numframe1,1);
% IW_PSNR = zeros(Numframe1,1);
ave_IW_SSIM = zeros(length(testvideo),1);
% ave_IW_PSNR = zeros(length(testvideo),1);

MS_SSIM = zeros(Numframe1,1);
ave_MS_SSIM = zeros(length(testvideo),1);

for i = 1:length(testvideo)
    disinputfile = ['E:\paper_work\DIBR_experiments\Distortion\H265\lovebird\Y\' testvideo{i} '\'];
    % disFiles = dir([disinputfile '*.png']);
    for j = 1:Numframe
        imgname = sprintf('denoised_Y%d.png',j-1);
        disImgPath =[disinputfile imgname];
        orgImgPath =[orgVideoPath orgFiles(j).name];
        imgA = imread(orgImgPath);
        imgB = imread(disImgPath);


        MP_PSNR_full(j) = mp_psnr( imgA, imgB );
        %MP_PSNR_reduc(j) = mp_psnr_reduc( imgA, imgB );


        %MS_SSIM(j) = msssim(imgA, imgB);
        
        [iw_ssim,iwmse,iw_psnr]= iwssim(imgA, imgB);
        IW_SSIM(j) = iw_ssim;
 
    end
    ave_MP_PSNR(i) = mean(MP_PSNR_full);
    %ave_MP_PSNR_reduc(i) = mean(MP_PSNR_reduc);
    %ave_MS_SSIM(i) = mean(MS_SSIM);
    ave_IW_SSIM(i) = mean(IW_SSIM);

   
    fidsc = fopen(saveFileName,'a+');
    fprintf(fidsc, '%s %f %f %f %f %f\r\n',testvideo{i},ave_MP_PSNR(i),ave_IW_SSIM(i));
    fclose(fidsc);
end



