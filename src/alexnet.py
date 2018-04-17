from __future__ import print_function
# based on https://github.com/jiecaoyu/pytorch_imagenet

import torch
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import scipy.misc
import torchvision.transforms
import numpy
from torch.autograd import Variable

__all__ = [
        'AlexNet',
        'load_caffe_net',
        'load_alexnet_from_caffe',
        'alexnet_from_caffe_net',
        'copy_alexnet_caffe_params',
        'copy_alexnet_to_caffe',
        'IMAGE_MEAN',
        'IMAGE_STDEV',
        ]

IMAGE_256_MEAN = numpy.array([123, 117, 104])
IMAGE_MEAN = (IMAGE_256_MEAN / 255.0)
IMAGE_STDEV = numpy.array([0.22, 0.22, 0.22])

class AlexNet(nn.Sequential):

    def __init__(self, input_channels=None, output_channels=None,
            first_layer=None, last_layer=None,
            include_lrn=True, split_groups=True,
            layer_sizes=None,
            extra_output=None,
            horizontal_flip=False,
            half_resolution=False,
            modify_sequence=None):
        self.horizontal_flip = horizontal_flip
        including = (first_layer == None)
        w = [3, 96, 256, 384, 384, 256, 4096, 4096, 365]
        if layer_sizes is not None:
            for layer_name in layer_sizes:
                w[int(layer_name[-1])] = layer_sizes[layer_name]
        if input_channels is not None:
            if first_layer is not None:
                w[int(first_layer[-1]) - 1] = input_channels
            else:
                w[0] = input_channels
        if output_channels is not None:
            if last_layer is not None:
                w[int(last_layer[-1])] = output_channels
            else:
                w[-1] = output_channels
        self.extra_output = extra_output
        if split_groups is True:
            groups = [1, 2, 1, 2, 2]
        elif not split_groups:
            groups = [1, 1, 1, 1, 1]
        else:
            groups = [1, 1, 1, 1, 1]
            for layer_name in split_groups:
                groups[int(layer_name[-1]) - 1] = split_groups[layer_name]
        if modify_sequence is None:
            modify_sequence = lambda x: x
        include = (first_layer == None)
        sequence = OrderedDict()
        for name, module in modify_sequence([
                ('conv1', nn.Conv2d(w[0], w[1], kernel_size=11,
                    stride=2 if half_resolution else 4,
                    groups=groups[0])),
                ('relu1', nn.ReLU(inplace=True)),
                ('pool1', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('lrn1', LRN(local_size=5, alpha=0.0001, beta=0.75)),
                ('conv2', nn.Conv2d(w[1], w[2], kernel_size=5, padding=2,
                    groups=groups[1])),
                ('relu2', nn.ReLU(inplace=True)),
                ('pool2', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('lrn2', LRN(local_size=5, alpha=0.0001, beta=0.75)),
                ('conv3', nn.Conv2d(w[2], w[3], kernel_size=3, padding=1,
                    groups=groups[2])),
                ('relu3', nn.ReLU(inplace=True)),
                ('conv4', nn.Conv2d(w[3], w[4], kernel_size=3, padding=1,
                    groups=groups[3])),
                ('relu4', nn.ReLU(inplace=True)),
                ('conv5', nn.Conv2d(w[4], w[5], kernel_size=3, padding=1,
                    groups=groups[4])),
                ('relu5', nn.ReLU(inplace=True)),
                ('pool5', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('flatten', Vectorize()),
                ('fc6', nn.Linear(w[5] * 6 * 6, w[6])),
                ('relu6', nn.ReLU(inplace=True)),
                ('dropout6', nn.Dropout()),
                ('fc7', nn.Linear(w[6], w[7])),
                ('relu7', nn.ReLU(inplace=True)),
                ('dropout7', nn.Dropout()),
                ('fc8', nn.Linear(w[7], w[8])) ]):
            if not include_lrn and name.startswith('lrn'):
                continue
            if name == first_layer:
                include = True
            if include:
                sequence[name] = module
            if name == last_layer:
                break
        if first_layer and first_layer not in sequence:
            raise ValueError('layer %s not found' % first_layer)
        if last_layer and last_layer not in sequence:
            raise ValueError('layer %s not found' % last_layer)
        super(AlexNet, self).__init__(sequence)
        # Create hooks to retain values for any queried layers
        self.retained = None

    def forward(self, input):
        if (self.horizontal_flip):
            input = flip(input, dim=3)
        extra = []
        for name, module in self._modules.items():
            input = module(input)
            if self.extra_output and name in self.extra_output:
                extra.append(input)
        if (self.horizontal_flip and len(input.shape) > 3):
            input = flip(input, dim=3)
        if len(extra):
            return (input,) + tuple(extra)
        return input

    def retain(self, layers):
        self.retained = {}
        self.retained_input = {}
        def make_hook(layer):
            def hook(module, input, output):
                self.retained[layer] = output.data.cpu().numpy()
                self.retained_input[layer] = input[0].data.cpu().numpy()
            getattr(self, layer).register_forward_hook(hook)
        for layer in layers:
            make_hook(layer)

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

class Vectorize(nn.Module):
    def __init__(self):
        super(Vectorize, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), int(numpy.prod(x.size()[1:])))
        return x

def get_caffe_param(net, layername, param):
    layerindex = list(net._layer_names).index(layername)
    paramindex = dict(
            weight=0, bias=1)[param]
    return net.layers[layerindex].blobs[paramindex].data

def set_caffe_param(net, layername, param, value):
    get_caffe_param(net, layername, param)[...] = value

def set_torch_param(model, layer, paramname, value, cuda=False):
    torchval = torch.from_numpy(value)
    if cuda:
        torchval = torchval.cuda()
    getattr(getattr(model, layer), paramname).data = torchval
    model.float()

def get_torch_param(model, layer, paramname):
    return getattr(getattr(model, layer), paramname).data.cpu().numpy()

def load_caffe_net(caffename):
    os.environ["GLOG_minloglevel"] = "2"
    import caffe
    caffe.set_mode_cpu()
    if isinstance(caffename, tuple):
        prototxt, caffemodel = caffename
    else:
        prototxt = '%s.prototxt' % caffename
        caffemodel = '%s.caffemodel' % caffename
    if not caffemodel:
        return caffe.Net(str(prototxt), caffe.TEST)
    else:
        return caffe.Net(str(prototxt), str(caffemodel), caffe.TEST)

def load_alexnet_from_caffe(caffename):
    cnet = load_caffe_net(caffename)
    return alexnet_from_caffe_net(cnet)

def alexnet_from_caffe_net(cnet):
    # Make a dict of all the net layers
    clayer = OrderedDict([(cnet._layer_names[i], cnet.layers[i])
        for i in range(len(cnet.layers))])
    # Read the number of output classes
    outblob = cnet.outputs[0]
    num_classes = cnet.blobs[outblob].shape[1]
    # Set up a pytorch alexnet
    tnet = AlexNet(output_channels=num_classes)
    copy_alexnet_caffe_params(tnet, cnet)
    return tnet

def copy_alexnet_caffe_params(tnet, cnet, cuda=False, truncate=False):
    layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
    for layer in layers:
        # The torch net may be truncated, so skip any missing layers.
        if not hasattr(tnet, layer):
            continue
        for param in ['weight', 'bias']:
            oldval = get_torch_param(tnet, layer, param)
            newval = get_caffe_param(cnet, layer, param)
            if truncate and tuple(oldval.shape) != tuple(newval.shape):
                newval = newval[tuple(slice(0, s) for s in oldval.shape)]
            assert tuple(oldval.shape) == tuple(newval.shape), '%s: %r' % (
                    layer, (oldval.shape, newval.shape))
            if (param, layer) == ('weight', 'conv1'):
                # Switch conv1 to RGB, with input 0 mean 1 stdev
                newval = (newval[:,[2,1,0]] *
                        255.0 * IMAGE_STDEV[None,:,None,None])
            set_torch_param(tnet, layer, param, newval, cuda=cuda)
    return tnet

def copy_alexnet_to_caffe(tnet, cnet):
    layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
    for layer in layers:
        for param in ['weight', 'bias']:
            newval = get_torch_param(tnet, layer, param)
            if (param, layer) == ('weight', 'conv1'):
                # Switch conv1 back to original scale
                newval = (newval[:,[2,1,0]] /
                        255.0 / IMAGE_STDEV[None,:,None,None])
            set_caffe_param(cnet, layer, param, newval)
    return cnet

# See https://github.com/pytorch/pytorch/issues/229#issuecomment-350041662
def flip(x, dim=-1):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

######################################################################
# TESTING CODE BELOW.
######################################################################

preprocess = torchvision.transforms.Normalize(IMAGE_MEAN, IMAGE_STDEV)

def caffe_preprocess(img):
    """Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
    out = numpy.copy(img)
    out -= IMAGE_256_MEAN[None, None, :]
    out = out[:, :, [2, 1, 0]]  # swap channel from RGB to BGR
    return out

def load_image(path, size=227):
    img = scipy.misc.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    resized_img = scipy.misc.imresize(crop_img, (size, size))
    return resized_img.astype('float')

def verify_equal(cnet, tnet, imgfile):
    from numpy.testing import assert_almost_equal
    img = load_image(imgfile, cnet.blobs['data'].data[0].shape[1])

    # Run caffe model
    img_p = caffe_preprocess(img)
    img_pc = img_p.transpose(2, 0, 1)
    cnet.blobs['data'].reshape(*((1, ) + img_pc.shape))
    cnet.blobs['data'].data[0] = img_pc
    cnet.forward()

    # Run pytorch model
    tnet.eval()
    tnet.retain([
        'relu1', 'relu2', 'relu3', 'relu4', 'relu5', 'relu6', 'relu7', 'fc8' ])
    img_u = img / 255.0
    inp = preprocess(torch.from_numpy(img_u.transpose((2, 0, 1)))
            ).float().unsqueeze(0)
    out = tnet(Variable(inp))
    computed = tnet.retained

    # Check equality
    assert_almost_equal(cnet.blobs['conv1'].data, computed['relu1'], decimal=3)
    assert_almost_equal(cnet.blobs['conv2'].data, computed['relu2'], decimal=3)
    assert_almost_equal(cnet.blobs['conv3'].data, computed['relu3'], decimal=3)
    assert_almost_equal(cnet.blobs['conv4'].data, computed['relu4'], decimal=4)
    assert_almost_equal(cnet.blobs['conv5'].data, computed['relu5'], decimal=4)
    assert_almost_equal(cnet.blobs['fc6'].data, computed['relu6'], decimal=5)
    assert_almost_equal(cnet.blobs['fc7'].data, computed['relu7'], decimal=5)
    assert_almost_equal(cnet.blobs['fc8'].data, computed['fc8'], decimal=5)
    return True

if __name__ == '__main__':
    if len(sys.argv) == 4:
        ctxt = sys.argv[1]
        cmod = sys.argv[2]
        tmod = sys.argv[3]
        print('converting caffe model {} (with prototxt {}) to pytorch model {}'.format(cmod, ctxt, tmod))
        tnet = load_alexnet_from_caffe((ctxt, cmod))
        print('loaded caffe model, saving')
        torch.save(tnet.state_dict(), tmod)
        print('done')
        sys.exit(0)
    print('to convert model instead of running test, use: python alexnet.py TXT MOD OUT')
    testinput = os.path.join(os.path.dirname(__file__), 'harbor.jpg')
    pop = 'alexnet_places365'
    cnet = load_caffe_net((
     'population/%s/%s.prototxt' % (pop, pop),
     'population/%s/%s_000_iter_420000.caffemodel' % (pop, pop)))
    tnet = alexnet_from_caffe_net(cnet)
    if verify_equal(cnet, tnet, testinput):
        print('Torch model matches caffe model.')
    cnet2 = load_caffe_net(('population/%s/%s.prototxt' % (pop, pop), None))
    copy_alexnet_to_caffe(tnet, cnet2)
    if verify_equal(cnet2, tnet, testinput):
        print('Caffe model matches torch model.')
