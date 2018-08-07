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

class WindowDoubleBackpropLoss(object):
    def __init__(self, beta):
        self.beta = beta

    def __call__(self, inp, out_with_extra, target):
        output = out_with_extra[0]
        features = out_with_extra[1:]
        loss = nn.functional.cross_entropy(output, target)
        # Now compute 2nd derivative penalty.
        # grad_features = torch.autograd.grad(loss, features, create_graph=True)
        # grad_norm = 0
        # for gf in grad_features:
        #     grad_norm = grad_norm + gf.pow(2).mean()
        # For each image, compute the most important feature
        grad_inp = 0
        f = features[0]
        max_features, max_chan_loc = f.view(f.shape[0], -1).max(1)
        summed_max_f = max_features.sum()
        (g_inp,) = torch.autograd.grad(summed_max_f, [inp], create_graph=True)
        # Only penalize gradient outside a window of where the max was.
        # This is the window trick.
        max_loc = max_chan_loc % (f.shape[2] * f.shape[3])
        max_mask = torch.ones(f.shape[0], 1, f.shape[2], f.shape[3],
                dtype=f.dtype, device=f.device)
        max_mask.view(f.shape[0], -1).scatter_(1, max_loc[:,None], 0)
        upsamp_mask = torch.ones_like(g_inp)
        upsamp_mask.view(g_inp.shape[0], g_inp.shape[1],
                f.shape[2], g_inp.shape[2] // f.shape[2],
                f.shape[3], g_inp.shape[3] // f.shape[3])[
                        ...] = max_mask[:,:,:,None,:,None]
        grad_inp = (g_inp * upsamp_mask).pow(2).sum()
        # Full loss
        inp_sens = self.beta * grad_inp
        # feat_sens = self.alpha * grad_norm
        regularized_loss = loss + inp_sens
        # print('loss vs grad norm: %g vs %g' % (loss, grad_norm))
        return regularized_loss, loss, inp_sens

def main():
    progress = default_progress()
    experiment_dir = 'experiment/win0_resnet_qcrop'
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
        batch_size=32, shuffle=True,
        num_workers=24, pin_memory=True)
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
    model = CustomResNet(18, num_classes=100, halfsize=True,
            extra_output=['layer3'])

    model.train()
    model.cuda()

    # An abbreviated training schedule: 40000 batches.
    # TODO: tune these hyperparameters.
    # init_lr = 0.002
    init_lr = 1e-4
    # max_iter = 40000 - 34.5% @1
    # max_iter = 50000 - 37% @1
    # max_iter = 80000 - 39.7% @1
    # max_iter = 100000 - 40.1% @1
    max_iter = 50000
    criterion = WindowDoubleBackpropLoss(1e0)
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
        # val_loss, val_acc = AverageMeter(), AverageMeter()
        val_acc = AverageMeter()
        for input, target in progress(val_loader):
            # Load data
            input_var, target_var = [d.cuda() for d in [input, target]]
            # Evaluate model
            with torch.no_grad():
                output = model(input_var)
                # loss, unreg_loss = criterion(output, target_var)
                _, pred = output[0].max(1)
                accuracy = (target_var.eq(pred)
                        ).data.float().sum().item() / input.size(0)
            # val_loss.update(loss.data.item(), input.size(0))
            val_acc.update(accuracy, input.size(0))
            # Check accuracy
            # post_progress(l=val_loss.avg, a=val_acc.avg*100.0)
            post_progress(a=val_acc.avg*100.0)
        # Save checkpoint
        save_checkpoint({
            'iter': iter_num,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'accuracy': val_acc.avg,
            # 'loss': val_loss.avg,
        }, val_acc.avg > best['val_accuracy'])
        best['val_accuracy'] = max(val_acc.avg, best['val_accuracy'])
        print_progress('Iteration %d val accuracy %.2f' %
                (iter_num, val_acc.avg * 100.0))

    # Here is our training loop.
    while iter_num < max_iter:
        for input, target in progress(train_loader):
            # Track the average training loss/accuracy for each epoch.
            train_loss, train_acc = AverageMeter(), AverageMeter()
            train_loss_u = AverageMeter()
            train_inp_sens = AverageMeter()
            # train_feat_sens = AverageMeter()
            # Load data
            input_var, target_var = [d.cuda() for d in [input, target]]
            # Evaluate model
            input_var.requires_grad = True
            output = model(input_var)
            loss, unreg_loss, inp_sens = criterion(
                    input_var, output, target_var)
            train_loss.update(loss.data.item(), input.size(0))
            train_loss_u.update(unreg_loss.data.item(), input.size(0))
            train_inp_sens.update(inp_sens.data.item(), input.size(0))
            # train_feat_sens.update(f_sens.data.item(), input.size(0))
            # Perform one step of SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Also check training set accuracy
            _, pred = output[0].max(1)
            accuracy = (target_var.eq(pred)).data.float().sum().item() / (
                    input.size(0))
            train_acc.update(accuracy)
            remaining = 1 - iter_num / float(max_iter)
            post_progress(u=train_loss_u.avg,
                    i=train_inp_sens.avg, a=train_acc.avg*100.0)
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
