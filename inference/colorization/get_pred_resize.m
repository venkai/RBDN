function get_pred_resize(new_size)

% Use this instead of get_pred.m for colorizing very high-res images.
% All results in the paper were however generated with get_pred.m.


if nargin < 1, new_size=[224, 224]; end;
if numel(new_size)==1, new_size=[new_size, new_size]; end;

config=create_config;
net = caffe.Net(config.model,config.weights,'test');
imgdir='../Data/colorization';
resdir='./results';
if ~isdir(resdir), mkdir(resdir); end
% Get image list
d=dir(strcat(imgdir,'/*'));
d={d.name}'; d=d(3:end);
for i=1:length(d)
    imgfile=d{i};
    dt=strfind(imgfile,'.'); dt=dt(end);
    resfile=strcat(resdir,'/',imgfile(1:dt),'png');
    I=imread(strcat(imgdir,'/',imgfile));
    if size(I,3)==1, I=repmat(I,[1,1,3]); end
    img_lab=rgb2lab(I);
    I_rz=rgb2lab(imresize(I,new_size,'bicubic'));
    img=permute(I_rz(:,:,1),[2,1,3])-50; % H*W -> W*H
    img=img(1:end-rem(end,32),1:end-rem(end,32));
    fprintf('Processing %s: [%d x %d]\n',imgfile,size(img,1),size(img,2));
    % Reshape blob 'data'
    net.blobs('data').reshape([size(img,1) size(img,2) 1 1]);
    net.reshape(); %Reshape remaining blobs accordingly
    cnn_input={single(img)};
    pred=net.forward(cnn_input);
    pred=imresize(permute(pred{1},[2,1,3]),[size(I,1),size(I,2)]);
    pred_lab=zeros(size(img_lab));
    pred_lab(:,:,1)=img_lab(:,:,1);
    pred_lab(:,:,2:3)=pred;
    pred_rgb=(lab2rgb(pred_lab));
    imwrite(pred_rgb,resfile);
end
caffe.reset_all();

function config=create_config
config.model='./test.prototxt';
config.weights='../../models/rbdn_colorization.caffemodel';
config.caffe_root = '../caffe_colorization';
fprintf('initializing caffe..\n');
addpath(fullfile(config.caffe_root, 'matlab'));
config.gpuNum=0; caffe.set_mode_gpu(); caffe.set_device(config.gpuNum);
%caffe.set_mode_cpu();
caffe.reset_all();
