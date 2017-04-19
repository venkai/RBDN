function get_pred

rng(9);
config=create_config;
config.resdir='./results/';
config.datadir='../Data/denoising';
if ~isdir(config.resdir), mkdir(config.resdir); end

fid=fopen(sprintf('%s/imgset/test.txt',config.datadir),'r');
imglist=textscan(fid,'%s%s'); fclose(fid);

% Note that the network is fully convolutional and can be reshaped to
% handle variable input size. Sort test images in descending order of size
% in order to minimize calls to net.reshape().
rs_indices=[];
load(sprintf('%s/imgset/imglist_dn_test_sorted.mat',config.datadir));
imglist=imglist{1}(rs_indices);


config.imglist=imglist;
config.net=load_net(config);
hcurr=512; wcurr=512;
for ind=1:1:length(imglist)
    fprintf('[%d/%d] %s\n',ind,length(imglist),imglist{ind});
    [hcurr,wcurr]=save_pred_oneimg(ind,config,hcurr,wcurr);
end
caffe.reset_all();
fprintf('\n\n------------\n');
fprintf('Computing Peak Signal to Noise Ratio for Results\n')
fprintf('------------\n\n');
compute_psnr;

function[hnew,wnew]=save_pred_oneimg(ind,config,h_old,w_old)
net=config.net;
imgrelpath=config.imglist{ind};
imgpath=sprintf('%s%s',config.datadir,imgrelpath);
resfile=sprintf('%s%s',config.resdir,imgrelpath);
% Try to maintain same directory structure for resdir as in datadir and
% create sub-directories in resdir as neccessary
sl_ind=strfind(resfile,'/'); sl_ind=sl_ind(end)-1;
resdir=resfile(1:sl_ind);
if ~isdir(resdir), mkdir(resdir); end
% Check if resfile exists
if exist(resfile,'file')==2
    fprintf('RESULT %s already exists\n',resfile);
    hnew=h_old; wnew=w_old;
    return;
end
% Do forward pass with image
I=imread(imgpath);
[h,w,~]=size(I);
% Note that for a K-branch RBDN, we have K+1 downsampling layers.
% So height and width must be a multiple of 2^(K+1).
% If this is not the case, we slide the network across the image
% and stitch (average) the predictions later.
fac=2^(config.branches+1);
sh=rem(h,fac); sw=rem(w,fac);
hnew=h-sh; wnew=w-sw;
O=zeros(size(I)); % Per-pixel sum of predictions
C=zeros(size(I)); % Per-pixel count of # of additions 
if h_old~=hnew || w_old~=wnew % Reshape only when required
    fprintf('Reshaping Network from [%dx%d] to [%dx%d]\n',h_old,w_old,hnew,wnew); 
    net.blobs('data').reshape([wnew hnew 1 1]); % Reshape blob 'data'
    net.reshape(); % Reshape remaining blobs accordingly
end
fprintf('Stitching %dx%d=%d dense reconstructions\n',sh+1,sw+1,(sh+1)*(sw+1)); 
for i=0:sh
    for j=0:sw
        img=I(1+i:hnew+i,1+j:wnew+j,:);
        img=permute(img,[2,1,3]); % H*W -> W*H
        cnn_input={single(img)};
        pred=net.forward(cnn_input);
        pred=permute(pred{1},[2,1,3]); % W*H -> H*W
        O(1+i:hnew+i,1+j:wnew+j,:)=O(1+i:hnew+i,1+j:wnew+j,:)+pred;
        C(1+i:hnew+i,1+j:wnew+j,:)=C(1+i:hnew+i,1+j:wnew+j,:)+1;
    end
end
% Take average over overlapping predictions
O=O./C;
O=uint8(O);
imwrite(O,resfile);

function net = load_net(config)

net = caffe.Net(config.model,config.weights,'test');
fprintf('\n\nModel: %s\nWeights: %s\n\n',config.model,config.weights);

function config=create_config
config.model='./test.prototxt';
config.weights='../../models/rbdn_denoising.caffemodel';
config.branches=3;

config.caffe_root = '../caffe';
fprintf('initializing caffe..\n');
addpath(fullfile(config.caffe_root, 'matlab'));
config.gpuNum=0; caffe.set_mode_gpu(); caffe.set_device(config.gpuNum);
% caffe.set_mode_cpu();
