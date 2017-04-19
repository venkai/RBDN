function get_pred

%--
% Create network and data config
config=create_config;
%--

% Prepare imagelist
d=dir(sprintf('%s/*',config.datadir));
d={d.name}; d=d(3:end);
config.imglist=d;
% Load RBDN network from config
config.net=load_net(config);
for ind=1:length(d)
    fprintf('[%d/%d] %s\n',ind,length(d),d{ind});
    save_pred_oneimg(ind,config);
end
caffe.reset_all();

function save_pred_oneimg(ind,config)
net=config.net;
imgrelpath=config.imglist{ind};
imgpath=sprintf('%s/%s',config.datadir,imgrelpath);
resfile=sprintf('%s/%s',config.resdir,strrep(imgrelpath,'.jpg','.png'));

% Do forward pass with image
img=imread(imgpath);
% Force color if bw
if size(img,3)==1, img=repmat(img,[1,1,3]); end
img=imresize(img,[224,224]);
% Convert RGB to BGR
img=img(:,:,[3,2,1]);
% Convert H*W to W*H
img=permute(img,[2,1,3]);
cnn_input={single(img)};
% Forward pass 
pred=net.forward(cnn_input);
% Convert W*H to H*W
pred=permute(pred{1},[2,1,3]);
% Convert BGR to RGB
pred=uint8(pred(:,:,[3,2,1]));
imwrite(pred,resfile);


function net = load_net(config)

model = './test.prototxt';
weights = '../../models/rbdn_relighting.caffemodel';
net = caffe.Net(model,weights,'test');

function config=create_config

config.datadir = '../Data/relighting';
config.resdir = './results';
if ~isdir(config.resdir), mkdir(config.resdir); end;
config.caffe_root = '../caffe';
fprintf('initializing caffe..\n');
addpath(fullfile(config.caffe_root, 'matlab'));
config.gpuNum=0; caffe.set_mode_gpu(); caffe.set_device(config.gpuNum);
%caffe.set_mode_cpu();
caffe.reset_all();