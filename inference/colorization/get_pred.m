function get_pred

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
    %if exist(resfile,'file')==2, continue; end;
    I=imread(strcat(imgdir,'/',imgfile));
    % If input image is larger that 800*800 (i.e. # of pixels > 640000),
    % then network doesn't fit in GPU memory. Of course this depends on how much
    % memory you have and if you are operating in CPU/GPU mode.
    if numel(I(:,:,1))>640000
      if size(I,1)>=size(I,2) 
        I=imresize(I,[800,NaN],'bicubic');
      else 
        I=imresize(I,[NaN,800],'bicubic');
      end
    end
    if size(I,3)==1, I=repmat(I,[1,1,3]); end
    %imshow(I);
    [h,w,~]=size(I);
    I=padarray(I,[64,64],'symmetric');
    I=I(1:end-rem(end,32),1:end-rem(end,32),:);
    img_lab=rgb2lab(I);
    img=permute(img_lab(:,:,1),[2,1,3])-50;
    fprintf('[%d/%d] Processing %s: [%d x %d]\n',i,length(d),imgfile,size(img,1),size(img,2));
    net.blobs('data').reshape([size(img,1) size(img,2) 1 1]); % reshape blob 'data'
    net.reshape(); %Reshape remaining blobs accordingly
    cnn_input={single(img)};
    pred=net.forward(cnn_input);
    pred=imresize(permute(pred{1},[2,1,3]),4);
    pred_lab=zeros(size(img_lab));
    pred_lab(:,:,1)=img_lab(:,:,1);
    pred_lab(:,:,2:3)=pred;
    pred_rgb=(lab2rgb(pred_lab));
    pred_rgb=pred_rgb(65:h+64,65:w+64,:);
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
