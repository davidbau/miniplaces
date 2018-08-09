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

def main():
    parser = argparse.ArgumentParser(description='Adversarial test')
    parser.add_argument('--expdir', type=str, default=None, required=True,
                        help='experiment directory containing model')
    args = parser.parse_args()
    progress = default_progress()
    experiment_dir = args.expdir
    perturbation1 = numpy.load('perturbation/VGG-19.npy')
    perturbation = numpy.load('perturbation/perturb_synth.npy')
    print('Original std %e new std %e' % (numpy.std(perturbation1),
        numpy.std(perturbation)))
    perturbation *= numpy.std(perturbation1) / numpy.std(perturbation)
    # To deaturate uncomment.
    # perturbation = numpy.repeat(perturbation[:,:,1:2], 3, axis=2)

    val_loader = torch.utils.data.DataLoader(
        CachedImageFolder('dataset/miniplaces/simple/val',
            transform=transforms.Compose([
                        transforms.Resize(128),
                        transforms.CenterCrop(112),
                        AddPerturbation(perturbation[48:160,48:160]),
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGE_MEAN, IMAGE_STDEV),
                        ])),
        batch_size=32, shuffle=False,
        num_workers=0, pin_memory=True)
    # Create a simplified ResNet with half resolution.
    model = CustomResNet(18, num_classes=100, halfsize=True)
    checkpoint_filename = 'best_miniplaces.pth.tar'
    best_checkpoint = os.path.join(experiment_dir, checkpoint_filename)
    checkpoint = torch.load(best_checkpoint)
    iter_num = checkpoint['iter']
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    val_loss, val_acc = AverageMeter(), AverageMeter()
    for input, target in progress(val_loader):
        # Load data
        input_var, target_var = [d.cuda() for d in [input, target]]
        # Evaluate model
        with torch.no_grad():
            output = model(input_var)
            loss = criterion(output, target_var)
            _, pred = output.max(1)
            accuracy = (target_var.eq(pred)
                    ).data.float().sum().item() / input.size(0)
        val_loss.update(loss.data.item(), input.size(0))
        val_acc.update(accuracy, input.size(0))
        # Check accuracy
        post_progress(l=val_loss.avg, a=val_acc.avg)
    print_progress('Loss %e, validation accuracy %.4f' %
            (val_loss.avg, val_acc.avg))
    with open(os.path.join(experiment_dir, 'adversarial_test.json'), 'w') as f:
        json.dump(dict(
            adversarial_acc=val_acc.avg,
            adversarial_loss=val_loss.avg), f)

class AddPerturbation(object):
    def __init__(self, perturbation):
        self.perturbation = perturbation

    def __call__(self, pic):
        npyimg = numpy.array(pic, numpy.uint8, copy=False
                ).astype(numpy.float32)
        npyimg += self.perturbation
        npyimg.clip(0, 255, npyimg)
        return npyimg / 255.0  # As a float it should be [0..1]

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
