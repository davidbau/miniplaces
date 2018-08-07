import torch, numpy, os, shutil, math, re
from progress import set_default_verbosity, print_progress, post_progress
from progress import default_progress, desc_progress
from files import ensure_dir_for
from alexnet import AlexNet
from torch.autograd import Variable
from torch import nn
from torch.nn import init
from imagefolder import CachedImageFolder
from alexnet import IMAGE_MEAN, IMAGE_STDEV
from torchvision import transforms
from torch.optim import Optimizer

def is_positive_param(name):
    return 'weight' in name and 'conv3' in name and 'fc8' not in name

def main():
    progress = default_progress()
    experiment_dir = 'experiment/positive'
    # Here's our data
    train_loader = torch.utils.data.DataLoader(
        CachedImageFolder('dataset/miniplaces/simple/train',
            transform=transforms.Compose([
                        transforms.Resize(128),
                        transforms.RandomCrop(119),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGE_MEAN, IMAGE_STDEV) ])),
        batch_size=64, shuffle=True,
        num_workers=6, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CachedImageFolder('dataset/miniplaces/simple/val',
            transform=transforms.Compose([
                        transforms.Resize(128),
                        transforms.CenterCrop(119),
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGE_MEAN, IMAGE_STDEV) ])),
        batch_size=512, shuffle=False,
        num_workers=6, pin_memory=True)
    # Create a simplified AlexNet with half resolution.
    model = AlexNet(first_layer='conv1', last_layer='fc8',
            layer_sizes=dict(fc6=2048, fc7=2048),
            output_channels=100, half_resolution=True,
            modify_sequence=add_scale_layers,
            include_lrn=False, split_groups=False).cuda()
    # Initialize weights by loading an old model.
    init_data = torch.load('experiment/miniplaces/best_miniplaces.pth.tar')
    model.load_state_dict(init_data['state_dict'], strict=False)
    for name, val in model.named_parameters():
        if 'weight' in name:
            init.kaiming_uniform_(val)
            if is_positive_param(name):
                with torch.no_grad():
                    val.abs_()
        elif 'scale' in name:
            init.uniform_(val, -0.1, 0.1)
            # init.constant_(val, 1)
        else:
            # Init positive bias in many layers to avoid dead neurons.
            assert 'bias' in name
            init.constant_(val, 0 if any(name.startswith(layer)
                    for layer in ['conv1', 'conv3', 'fc8']) else 1)
    # An abbreviated training schedule: 40000 batches.
    # TODO: tune these hyperparameters.
    # init_lr = 0.002
    init_lr = 0.002
    # max_iter = 40000 - 34.5% @1
    # max_iter = 50000 - 37% @1
    # max_iter = 80000 - 39.7% @1
    # max_iter = 100000 - 40.1% @1
    max_iter = 100000
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
            model.parameters(),
            lr=init_lr,
            momentum=0.9, # 0.9,
            # weight_decay=0.001)
            weight_decay=0)
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
            input_var, target_var = [Variable(d.cuda(non_blocking=True))
                    for d in [input, target]]
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
            input_var, target_var = [Variable(d.cuda(non_blocking=True))
                    for d in [input, target]]
            # Evaluate model
            output = model(input_var)
            loss = criterion(output, target_var)
            train_loss.update(loss.data.item(), input.size(0))
            # Perform one step of SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Project to positive weights only
            if True:
                for name, val in model.named_parameters():
                    if is_positive_param(name):
                        with torch.no_grad():
                            val.clamp_(min=0)
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

class ScaleLayer(nn.Module):
    def __init__(self, size):
        super(ScaleLayer, self).__init__()
        self.scale = nn.Parameter(torch.ones(size))

    def forward(self, x):
        uscale = self.scale.view(*((1, -1) + (1,) * (len(x.shape) - 2)))
        return x * uscale

def add_scale_layers(sequence):
    result = []
    for name, layer in sequence:
        result.append((name, layer))
        if is_positive_param(name):
            layernum = int(re.search('\d+', name).group(0))
            units = layer.out_features if 'fc' in name else layer.out_channels
            result.append(('scale%d' % (layernum), ScaleLayer(units)))
    return result

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
