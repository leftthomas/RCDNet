clear all;
ts =0;
tp =0;
for i=1:100                          % the number of testing samples
   x_true=im2double(imread(strcat('rain100L/norain/',sprintf('norain-%d.png',i))));  % groundtruth
   x_true = rgb2ycbcr(x_true);
   x_true = x_true(:,:,1); 
   x = im2double(imread(strcat('rain100L/derain/',sprintf('norain-%d.png',i))));     %reconstructed image
   x = rgb2ycbcr(x);
   x = x(:,:,1);
   tp= tp+ psnr(x,x_true);
   ts= ts+ssim(x*255,x_true*255);
end
fprintf('psnr=%6.4f, ssim=%6.4f\n',tp/100,ts/100)



