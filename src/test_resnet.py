import torch, numpy, os, shutil, math, re
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
    progress = default_progress()
    experiment_dir = 'experiment/resnet'
    val_loader = torch.utils.data.DataLoader(
        CachedImageFolder('dataset/miniplaces/simple/val',
            transform=transforms.Compose([
                        transforms.Resize(128),
                        # transforms.CenterCrop(112),
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGE_MEAN, IMAGE_STDEV),
                        ])),
        batch_size=32, shuffle=False,
        num_workers=24, pin_memory=True)
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
