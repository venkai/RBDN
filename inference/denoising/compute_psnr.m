function mPSNRs=compute_psnr

mPSNRs=zeros(1,11);
curr=0;
resdir='./psnr/';
if ~isdir(resdir), mkdir(resdir); end
for nlevel=10:5:60
    resfile=sprintf('%s/n%d_psnr.mat',resdir,nlevel);
    if exist(resfile,'file')==2,
        load(resfile,'mPSNR');
    else
        mPSNR=tmp_psnr_fixnoise(nlevel);
    end
    curr=curr+1;
    mPSNRs(curr)=mPSNR;
    fprintf('Test Noise std %0.2d: mPSNR=%0.2f\n',nlevel,mPSNR);
end
save(strcat(resdir,'/mPSNR.mat'),'mPSNRs');
fprintf('ns=%0.2d\t',10:5:60); fprintf('\n');
fprintf('%0.2f\t',mPSNRs); fprintf('\n');

function mPSNR=tmp_psnr_fixnoise(nlevel)

if nlevel<=25, ntype='low_noise'; else ntype='high_noise'; end
preddir=sprintf('./results/%s/test/noise_std_%d/noisy_image',ntype,nlevel);
gtdir=sprintf('../Data/denoising/%s/test/clean_images/clean_image',ntype);
% inputdir=sprintf('../Data/denoising/%s/test/noise_std_%d/noisy_image',ntype,nlevel);
resfile=sprintf('./psnr/n%d_psnr.mat',nlevel);
PSNR=zeros(300,1);
for i=1:300
    GT=imread(sprintf('%s_%d.png',gtdir,i));
    P=imread(sprintf('%s_%d.png',preddir,i));
    % I=imread(sprintf('%s_%d.png',inputdir,i));
    PSNR(i,1)=psnr(P,GT);
    % PSNR(i,2)=psnr(I,GT);
end
mPSNR=mean(PSNR);
save(resfile,'PSNR','mPSNR');