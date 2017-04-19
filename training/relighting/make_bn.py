
### User Configuration

iteration = 475
gpu_id_num = 0

## path configuration
caffe_root = '../caffe'
script_path = '.'
caffe_model = script_path + '/make_bn.prototxt'
caffe_weight = script_path + '/train_curr.caffemodel'
caffe_inference_weight = script_path + '/train_curr_inference.caffemodel'

## modify this definition according to model definition
bn_blobs= ['conv1',
           'convB1_1',
           'convB2_1',
           'convB3_1',
           'convB3_2',
           'deconvB3_1',
           'convB2_2',
           'deconvB2_1',
           'convB1_2',
           'deconvB1_1',
           'conv2_1',
           'conv2_2',
           'conv2_3',
           'conv2_4',
           'conv2_5',
           'conv2_6',
           'conv2_7',
           'conv2_8',
           'conv2_9']
bn_layers=['bn1',
           'bnB1_1',
           'bnB2_1',
           'bnB3_1',
           'bnB3_2',
           'debnB3_1',
           'bnB2_2',
           'debnB2_1',
           'bnB1_2',
           'debnB1_1',
           'bn2_1',
           'bn2_2',
           'bn2_3',
           'bn2_4',
           'bn2_5',
           'bn2_6',
           'bn2_7',
           'bn2_8',
           'bn2_9']
bn_means= ['bn1-mean',
           'bnB1_1-mean',
           'bnB2_1-mean',
           'bnB3_1-mean',
           'bnB3_2-mean',
           'debnB3_1-mean',
           'bnB2_2-mean',
           'debnB2_1-mean',
           'bnB1_2-mean',
           'debnB1_1-mean',
           'bn2_1-mean',
           'bn2_2-mean',
           'bn2_3-mean',
           'bn2_4-mean',
           'bn2_5-mean',
           'bn2_6-mean',
           'bn2_7-mean',
           'bn2_8-mean',
           'bn2_9-mean']
bn_vars = ['bn1-var',
           'bnB1_1-var',
           'bnB2_1-var',
           'bnB3_1-var',
           'bnB3_2-var',
           'debnB3_1-var',
           'bnB2_2-var',
           'debnB2_1-var',
           'bnB1_2-var',
           'debnB1_1-var',
           'bn2_1-var',
           'bn2_2-var',
           'bn2_3-var',
           'bn2_4-var',
           'bn2_5-var',
           'bn2_6-var',
           'bn2_7-var',
           'bn2_8-var',
           'bn2_9-var']


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

def forward_once(net):
    start_ind = 0
    end_ind = len(net.layers) - 1
    net._forward(start_ind, end_ind)
    return {out: net.blobs[out].data for out in net.outputs}

print net.params.keys()

res = forward_once(net)

bn_avg_mean = {bn_mean: np.squeeze(res[bn_mean]).copy() for bn_mean in bn_means}
bn_avg_var = {bn_var: np.squeeze(res[bn_var]).copy() for bn_var in bn_vars}    

cnt = 1

for i in range(0, iteration):
    res = forward_once(net)
    for bn_mean in bn_means:
        bn_avg_mean[bn_mean] = bn_avg_mean[bn_mean] + np.squeeze(res[bn_mean])
    for bn_var in bn_vars:
        bn_avg_var[bn_var] = bn_avg_var[bn_var] + np.squeeze(res[bn_var])
        
    cnt += 1
    print 'progress: %d/%d' % (i, iteration)

## compute average
for bn_mean in bn_means:
    bn_avg_mean[bn_mean] /= cnt
for bn_var in bn_vars:
    bn_avg_var[bn_var] /= cnt

for i in range(0, len(bn_vars)):
    m = np.prod(net.blobs[bn_blobs[i]].data.shape) / np.prod(bn_avg_var[bn_vars[i]].shape)
    bn_avg_var[bn_vars[i]] *= (m/(m-1))

scale_data = {bn_layer: np.squeeze(net.params[bn_layer][0].data) for bn_layer in bn_layers}
shift_data = {bn_layer: np.squeeze(net.params[bn_layer][1].data) for bn_layer in bn_layers}

var_eps = 1e-9

new_scale_data = {}
new_shift_data = {}
for i in range(0, len(bn_layers)):
    gamma = scale_data[bn_layers[i]]
    beta = shift_data[bn_layers[i]]
    Ex = bn_avg_mean[bn_means[i]]
    Varx = bn_avg_var[bn_vars[i]]
    new_gamma = gamma / np.sqrt(Varx + var_eps)
    new_beta = beta - (gamma * Ex / np.sqrt(Varx + var_eps))
    
    new_scale_data[bn_layers[i]] = new_gamma
    new_shift_data[bn_layers[i]] = new_beta

print new_scale_data.keys()
print new_shift_data.keys()

## assign computed new scale and shift values to net.params
for i in range(0, len(bn_layers)):
    net.params[bn_layers[i]][0].data[...] = new_scale_data[bn_layers[i]].reshape(net.params[bn_layers[i]][0].data.shape)
    net.params[bn_layers[i]][1].data[...] = new_shift_data[bn_layers[i]].reshape(net.params[bn_layers[i]][1].data.shape)

print 'start saving model'

net.save(caffe_inference_weight)

print 'done'



