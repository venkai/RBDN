
### User Configuration

iteration = 1000
gpu_id_num = 0

## path configuration
caffe_root = '../caffe_colorization'
script_path = '.'
caffe_model = script_path + '/test.prototxt'
caffe_weight = script_path + '/train_curr_inference.caffemodel'
caffe_inference_weight = script_path + '/train_curr_inference.caffemodel'


### start generate caffemodel

print 'start generating BN-testable caffemodel'
print 'caffe_root: %s' % caffe_root
print 'script_path: %s' % script_path
print 'caffe_model: %s' % caffe_model
print 'caffe_weight: %s' % caffe_weight
print 'caffe_inference_weight: %s' % caffe_inference_weight

import numpy as np

import sys
sys.path.append(caffe_root+'/python')
import caffe
from caffe.proto import caffe_pb2

# caffe.set_mode_cpu()
caffe.set_mode_gpu()
caffe.set_device(gpu_id_num)
net = caffe.Net(caffe_model, caffe_weight, caffe.TEST)
pts_in_hull = np.load('./resources/pts_in_hull.npy') # load cluster centers
net.params['class_ab'][0].data[:,:,0,0] = pts_in_hull.transpose((1,0)) # populate cluster centers as 1x1 convolution kernel
print 'Annealed-Mean Parameters populated'
print net.params.keys()
print 'start saving model'

net.save(caffe_inference_weight)

print 'done'



