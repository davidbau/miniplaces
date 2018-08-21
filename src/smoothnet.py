import numpy, torch
from customnet import CustomResNet
from torch import nn

class ConstantModuleWrapper(nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = [wrapped] # Wrapped module is not a submodule!
    def forward(self, *args):
        return self.wrapped[0].forward(*args)
    def _apply(self, fn):
        super()._apply(fn)
        self.wrapped[0]._apply(fn)

def make_smoother(channels):
    smoother = nn.Conv2d(channels, channels, 3, 1, bias=False)
    kernel = torch.from_numpy(numpy.array([
        [0.0625, 0.125, 0.0625],
        [0.125,  0.25,  0.125 ],
        [0.0625, 0.125, 0.0625]], dtype=numpy.float32))
    with torch.no_grad():
        smoother.weight.zero_()
        for c in range(channels):
            smoother.weight[c, c].copy_(kernel)
            # smoother.weight[c, c, 1, 1] = 1
        smoother.weight.requires_grad = False
    return ConstantModuleWrapper(smoother)

class Smoother(nn.Module):
    def __init__(self, stride=1):
        super().__init__()
        smoother = nn.Conv2d(1, 1, kernel_size=3, padding=1,
                stride=stride, bias=False)
        kernel = torch.from_numpy(numpy.array([
            [0.0625, 0.125, 0.0625],
            [0.125,  0.25,  0.125 ],
            [0.0625, 0.125, 0.0625]], dtype=numpy.float32))
        with torch.no_grad():
            smoother.weight.zero_()
            smoother.weight[0, 0].copy_(kernel)
            smoother.weight.requires_grad = False
        self.wrapper = [smoother]
    def forward(self, data):
        smoother = self.wrapper[0]
        shape = data.shape
        flattened = data.view(shape[0] * shape[1], 1, shape[2], shape[3])
        smoothed = smoother(flattened)
        unflattened = smoothed.view(shape[0], shape[1],
                smoothed.shape[2], smoothed.shape[3])
        return unflattened
    def _apply(self, fn):
        super()._apply(fn)
        self.wrapper[0]._apply(fn)

def add_input_smoother(sequence):
    sequence.insert(0, ('smoother', Smoother()))
    return sequence
    
class SmoothedResNet18(CustomResNet):
    def __init__(self):
        super().__init__(18, num_classes=100, halfsize=True,
                modify_sequence=add_input_smoother)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Smoother(stride=2),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

if __name__ == '__main__':
    SmoothedResNet18()
