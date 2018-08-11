import torch, numpy, os, shutil, math, re, argparse, json
from progress import set_default_verbosity, print_progress, post_progress
from progress import default_progress, desc_progress
from files import ensure_dir_for
from torch import nn
from torch.nn import init
from imagefolder import CachedImageFolder
from alexnet import IMAGE_MEAN, IMAGE_STDEV
from torchvision import transforms
from torch.optim import Optimizer
from customnet import CustomResNet
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description='Adversarial test')
    parser.add_argument('--expdir', type=str, default='experiment/resnet',
                        help='experiment directory containing model')
    args = parser.parse_args()
    progress = default_progress()
    experiment_dir = args.expdir
    perturbations = [
        numpy.zeros((224, 224, 3), dtype='float32'),
        numpy.load('perturbation/VGG-19.npy'),
        numpy.load('perturbation/perturb_synth.npy'),
        numpy.load('perturbation/perturb_synth_histmatch.npy'),
        numpy.load('perturbation/perturb_synth_newspectrum.npy'),
        numpy.load('perturbation/perturbation_rotated.npy'),
        numpy.load('perturbation/perturbation_rotated_averaged.npy'),
        numpy.load('perturbation/perturbation_noisefree.npy'),
        numpy.load('perturbation/perturbation_noisefree_nodc.npy'),
    ]
    perturbation_name = [
        "unattacked",
        "universal adversary",
        "synthetic",
        "histmatch",
        "newspectrum",
        "rotated",
        "rotated_averaged",
        "noisefree",
        "noisefree_nodc",
    ]
    print('Original std %e new std %e' % (numpy.std(perturbations[1]),
        numpy.std(perturbations[2])))
    perturbations[2] *= (
            numpy.std(perturbations[1]) / numpy.std(perturbations[2]))
    # To deaturate uncomment.
    loaders = [torch.utils.data.DataLoader(
            CachedImageFolder('dataset/miniplaces/simple/val',
                transform=transforms.Compose([
                            transforms.Resize(128),
                            # transforms.CenterCrop(112),
                            AddPerturbation(perturbation[40:168,40:168]),
                            transforms.ToTensor(),
                            transforms.Normalize(IMAGE_MEAN, IMAGE_STDEV),
                            ])),
            batch_size=32, shuffle=False,
            num_workers=0, pin_memory=True)
        for perturbation in perturbations]
    # Create a simplified ResNet with half resolution.
    model = CustomResNet(18, num_classes=100, halfsize=True)
    layernames = ['relu', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']
    retain_layers(model, layernames)
    checkpoint_filename = 'best_miniplaces.pth.tar'
    best_checkpoint = os.path.join(experiment_dir, checkpoint_filename)
    checkpoint = torch.load(best_checkpoint)
    iter_num = checkpoint['iter']
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    val_acc = [AverageMeter() for _ in loaders]
    diffs, maxdiffs, signflips = [
            [defaultdict(AverageMeter) for _ in perturbations]
            for _ in [1,2,3]]
    for all_batches in progress(zip(*loaders), total=len(loaders[0])):
        # Load data
        indata = [b[0].cuda() for b in all_batches]
        target = all_batches[0][1].cuda()
        # Evaluate model
        retained = defaultdict(list)
        with torch.no_grad():
            for i, inp in enumerate(indata):
                output = model(inp)
                _, pred = output.max(1)
                accuracy = (target.eq(pred)
                        ).data.float().sum().item() / inp.size(0)
                val_acc[i].update(accuracy, inp.size(0))
                for layer, data in model.retained.items():
                    retained[layer].append(data)
        for layer, vals in retained.items():
            for i in range(1, len(indata)):
                diffs[i][layer].update(
                        (vals[i] - vals[0]).pow(2).mean().item(),
                        len(target))
                maxdiffs[i][layer].update(
                        (vals[i] - vals[0]).view(len(target), -1
                            ).max(1)[0].mean().item(),
                        len(target))
                signflips[i][layer].update(
                        ((vals[i] > 0).float() - (vals[0] > 0).float()
                            ).abs().mean().item(),
                        len(target))
        # Check accuracy
        post_progress(a=val_acc[0].avg)
    # Report on findings
    for i, acc in enumerate(val_acc):
        print_progress('Test #%d (%s), validation accuracy %.4f' %
            (i, perturbation_name[i], acc.avg))
        if i > 0:
            for layer in layernames:
                print_progress(
                   'Layer %s RMS diff %.3e maxdiff %.3e signflip %.3e' %
                        (layer, math.sqrt(diffs[i][layer].avg),
                            maxdiffs[i][layer].avg, signflips[i][layer].avg))

class AddPerturbation(object):
    def __init__(self, perturbation):
        self.perturbation = perturbation

    def __call__(self, pic):
        npyimg = numpy.array(pic, numpy.uint8, copy=False
                ).astype(numpy.float32)
        npyimg += self.perturbation
        npyimg.clip(0, 255, npyimg)
        return npyimg / 255.0  # As a float it should be [0..1]

def retain_layer_output(dest, layer, name):
    '''Callback function to keep a reference to a layer's output in a dict.'''
    def hook_fn(m, i, output):
        dest[name] = output.detach()
    layer.register_forward_hook(hook_fn)

def retain_layers(model, layer_names):
    '''
    Creates a 'retained' property on the model which will keep a record
    of the layer outputs for the specified layers.  Also computes the
    cumulative scale and offset for convolutions.

    The layer_names array should be a list of layer names, or tuples
    of (name, aka) where the name is the pytorch name for the layer,
    and the aka string is the name you wish to use for the dissection.
    '''
    model.retained = {}
    seen = set()
    sequence = []
    aka_map = {}
    for name in layer_names:
        aka = name
        if not isinstance(aka, str):
            name, aka = name
        aka_map[name] = aka
    for name, layer in model.named_modules():
        sequence.append(layer)
        if name in aka_map:
            seen.add(name)
            aka = aka_map[name]
            retain_layer_output(model.retained, layer, aka)
    for name in aka_map:
        assert name in seen, ('Layer %s not found' % name)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    set_default_verbosity(True)
    main()
