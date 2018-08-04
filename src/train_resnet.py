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

class GlobalAveragePool2d(torch.nn.Module):
    def __init__(self):
        super(GlobalAveragePool2d, self).__init__()

    def forward(self, x):
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        return x

def main():
    progress = default_progress()
    experiment_dir = 'experiment/resnet'
    # Here's our data
    train_loader = torch.utils.data.DataLoader(
        CachedImageFolder('dataset/miniplaces/simple/train',
            transform=transforms.Compose([
                        transforms.Resize(128),
                        transforms.RandomCrop(112),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGE_MEAN, IMAGE_STDEV),
                        ])),
        batch_size=64, shuffle=True,
        num_workers=24, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CachedImageFolder('dataset/miniplaces/simple/val',
            transform=transforms.Compose([
                        transforms.Resize(128),
                        transforms.CenterCrop(112),
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGE_MEAN, IMAGE_STDEV),
                        ])),
        batch_size=128, shuffle=False,
        num_workers=24, pin_memory=True)
    # Create a simplified ResNet with half resolution.
    model = CustomResNet(18, num_classes=100, halfsize=True)

    model.train()
    model.cuda()

    # An abbreviated training schedule: 40000 batches.
    # TODO: tune these hyperparameters.
    # init_lr = 0.002
    init_lr = 0.0002
    # max_iter = 40000 - 34.5% @1
    # max_iter = 50000 - 37% @1
    # max_iter = 80000 - 39.7% @1
    # max_iter = 100000 - 40.1% @1
    max_iter = 10000
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters())
    iter_num = 0
    best = dict(val_accuracy=0.0)
    model.train()
    # Oh, hold on.  Let's actually resume training if we already have a model.
    checkpoint_filename = 'miniplaces.pth.tar'
    best_filename = 'best_%s' % checkpoint_filename
    best_checkpoint = os.path.join(experiment_dir, best_filename)
    try_to_resume_training = False
    if try_to_resume_training and os.path.exists(best_checkpoint):
        checkpoint = torch.load(os.path.join(experiment_dir, best_filename))
        iter_num = checkpoint['iter']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best['val_accuracy'] = checkpoint['accuracy']

    def save_checkpoint(state, is_best):
        filename = os.path.join(experiment_dir, checkpoint_filename)
        ensure_dir_for(filename)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename,
                    os.path.join(experiment_dir, best_filename))

    def validate_and_checkpoint():
        model.eval()
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
        # Save checkpoint
        save_checkpoint({
            'iter': iter_num,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'accuracy': val_acc.avg,
            'loss': val_loss.avg,
        }, val_acc.avg > best['val_accuracy'])
        best['val_accuracy'] = max(val_acc.avg, best['val_accuracy'])
        post_progress(v=val_acc.avg)

    # Here is our training loop.
    while iter_num < max_iter:
        for input, target in progress(train_loader):
            # Track the average training loss/accuracy for each epoch.
            train_loss, train_acc = AverageMeter(), AverageMeter()
            # Load data
            input_var, target_var = [d.cuda() for d in [input, target]]
            # Evaluate model
            output = model(input_var)
            loss = criterion(output, target_var)
            train_loss.update(loss.data.item(), input.size(0))
            # Perform one step of SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Also check training set accuracy
            _, pred = output.max(1)
            accuracy = (target_var.eq(pred)).data.float().sum().item() / (
                    input.size(0))
            train_acc.update(accuracy)
            remaining = 1 - iter_num / float(max_iter)
            post_progress(l=train_loss.avg, a=train_acc.avg,
                    v=best['val_accuracy'])
            # Advance
            iter_num += 1
            if iter_num >= max_iter:
                break
            # Linear learning rate decay
            lr = init_lr * remaining
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            # Ocassionally check validation set accuracy and checkpoint
            if iter_num % 1000 == 0:
                validate_and_checkpoint()
                model.train()

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
