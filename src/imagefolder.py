'''
Variants of pytorch's ImageFolder for loading image datasets with more
information, such as parallel feature channels in separate files,
cached files with lists of filenames, etc.
'''

import os, torch, re
import torch.utils.data as data
from torchvision.datasets.folder import default_loader
from PIL import Image
from progress import default_progress
from collections import OrderedDict

def grayscale_loader(path):
    with open(path, 'rb') as f:
        return Image.open(f).convert('L')

class FeatureFolder(data.Dataset):
    """
    A data loader that looks for parallel image filenames

    photo/park/004234.jpg
    photo/park/004236.jpg
    photo/park/004237.jpg

    feature/park/004234.png
    feature/park/004236.png
    feature/park/004237.png
    """
    def __init__(self, source_root, target_root,
            source_transform=None, target_transform=None,
            source_loader=default_loader, target_loader=grayscale_loader,
            verbose=None):
        self.imagepairs = make_feature_dataset(source_root, target_root,
                verbose=verbose)
        if len(self.imagepairs) == 0:
            raise RuntimeError("Found 0 images within: %s" % source_root)
        self.root = source_root
        self.target_root = target_root
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.source_loader = source_loader
        self.target_loader = target_loader

    def __getitem__(self, index):
        path, target_path = self.imagepairs[index]
        source = self.source_loader(path)
        target = self.target_loader(target_path)
        if self.source_transform is not None:
            source = self.source_transform(source)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return source, target

    def __len__(self):
        return len(self.imagepairs)

class FeatureAndClassFolder(data.Dataset):
    """
    A data loader that looks for parallel image filenames

    photo/park/004234.jpg
    photo/park/004236.jpg
    photo/park/004237.jpg

    feature/park/004234.png
    feature/park/004236.png
    feature/park/004237.png
    """
    def __init__(self, source_root, target_root,
            source_transform=None, target_transform=None,
            source_loader=default_loader, target_loader=grayscale_loader,
            verbose=None):
        classes, class_to_idx = find_classes(source_root)
        self.imagetriples= make_triples(source_root, target_root, class_to_idx,
                verbose=verbose)
        if len(self.imagetriples) == 0:
            raise RuntimeError("Found 0 images within: %s" % source_root)
        self.root = source_root
        self.target_root = target_root
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.source_loader = source_loader
        self.target_loader = target_loader

    def __getitem__(self, index):
        path, classidx, target_path = self.imagetriples[index]
        source = self.source_loader(path)
        target = self.target_loader(target_path)
        if self.source_transform is not None:
            source = self.source_transform(source)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return source, (classidx, target)

    def __len__(self):
        return len(self.imagetriples)

class ParallelImageFolders(data.Dataset):
    """
    A data loader that looks for parallel image filenames, for example

    photo1/park/004234.jpg
    photo1/park/004236.jpg
    photo1/park/004237.jpg

    photo2/park/004234.png
    photo2/park/004236.png
    photo2/park/004237.png
    """
    def __init__(self, image_roots,
            transform=None,
            loader=default_loader,
            stacker=torch.stack,
            intersection=False,
            verbose=None):
        classes, class_to_idx = find_classes(image_roots[0])
        self.images = make_parallel_dataset(image_roots, class_to_idx,
                intersection=intersection, verbose=verbose)
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images within: %s" % source_root)
        self.roots = image_roots
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.stacker = stacker
        self.loader = loader

    def __getitem__(self, index):
        paths, classidx = self.images[index]
        sources = [self.loader(path) for path in paths]
        if self.transform is not None:
            sources = [self.transform(source) for source in sources]
        if self.stacker is not None:
            sources = self.stacker(sources)
        return sources, classidx

    def __len__(self):
        return len(self.images)

class CachedImageFolder(data.Dataset):
    """
    A version of torchvision.dataset.ImageFolder that takes advantage
    of cached filename lists.

    photo/park/004234.jpg
    photo/park/004236.jpg
    photo/park/004237.jpg
    """
    def __init__(self, root,
            transform=None,
            loader=default_loader,
            verbose=None):
        classes, class_to_idx = find_classes(root)
        self.imgs = make_class_dataset(root, class_to_idx, verbose=verbose)
        if len(self.imgs) == 0:
            raise RuntimeError("Found 0 images within: %s" % root)
        self.root = root
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, classidx = self.imgs[index]
        source = self.loader(path)
        if self.transform is not None:
            source = self.transform(source)
        return source, classidx

    def __len__(self):
        return len(self.imgs)

    def filenames(self, indices=None, dirdepth=None):
        def normalize(fn):
            if dirdepth is None:
                return fn
            return os.path.join(*(splitpath(fn)[-1-dirdepth:]))
        if indices is None:
            indices = range(len(self.imgs))
        return [normalize(self.imgs[i][0]) for i in indices]

def splitpath(path):
    head, tail = os.path.split(path)
    if not head or head == path:
        return [head or tail]
    return splitpath(head) + [tail]

class StackFeatureChannels(object):
    def __init__(self, channels=None, keep_only=None):
        self.channels = channels
        self.keep_only = keep_only
    def __call__(self, tensor):
        if self.channels:
            channels = self.channels
            height = tensor.shape[1] // channels
        else:
            height = tensor.shape[2]
            channels = tensor.shape[1] // height
        result = tensor.view(channels, height, tensor.shape[2])
        if self.keep_only:
            result = result[:self.keep_only,...]
        return result

class SoftExpScale(object):
    def __init__(self, alpha=45.0):
        self.scale = 255.0 / alpha
    def __call__(self, tensor):
        return (tensor * self.scale).exp_().sub_(1)

def is_npy_file(path):
    return path.endswith('.npy') or path.endswith('.NPY')

def is_image_file(path):
    return None != re.match(r'.(jpe?g|png)$', path, re.IGNORECASE)

def walk_image_files(rootdir, verbose=None):
    progress = default_progress(verbose)
    indexfile = '%s.txt' % rootdir
    if os.path.isfile(indexfile):
        basedir = os.path.dirname(rootdir)
        with open(indexfile) as f:
            result = sorted([os.path.join(basedir, line.strip())
                for line in progress(f.readlines(),
                    desc='Reading %s' % os.path.basename(indexfile))])
            return result
    result = []
    for dirname, _, fnames in sorted(progress(os.walk(rootdir),
            desc='Walking %s' % os.path.basename(rootdir))):
        for fname in sorted(fnames):
            if is_image_file(fname) or is_npy_file(fname):
                result.append(os.path.join(dirname, fname))
    return result

def find_classes(dir):
	classes = [d for d in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, d))]
	classes.sort()
	class_to_idx = {classes[i]: i for i in range(len(classes))}
	return classes, class_to_idx

def make_feature_dataset(source_root, target_root, verbose=None):
    """
    Finds images in the subdirectories under source_root, and looks for
    similarly-located images (with the same directory structure
    and base filenames, but with possibly different file extensions)
    under target_root.  Each source image have a corresponding
    target image.
    """
    source_root = os.path.expanduser(source_root)
    target_root = os.path.expanduser(target_root)
    target_images = {}
    for path in walk_image_files(target_root, verbose=verbose):
        key = os.path.splitext(os.path.relpath(path, target_root))[0]
        target_images[key] = path
    imagepairs = []
    for path in walk_image_files(source_root, verbose=verbose):
        key = os.path.splitext(os.path.relpath(path, source_root))[0]
        if key not in target_images:
            raise RuntimeError('%s has no matching target %s.*' %
                    (path, os.path.join(target_root, key)) )
        imagepairs.append((path, target_images[key]))
    return imagepairs

def make_triples(source_root, target_root, class_to_idx, verbose=None):
    """
    Returns (source, classnum, feature)
    """
    source_root = os.path.expanduser(source_root)
    target_root = os.path.expanduser(target_root)
    target_images = {}
    for path in walk_image_files(target_root, verbose=verbose):
        key = os.path.splitext(os.path.relpath(path, target_root))[0]
        target_images[key] = path
    imagetriples = []
    for path in walk_image_files(source_root, verbose=verbose):
        key = os.path.splitext(os.path.relpath(path, source_root))[0]
        if key not in target_images:
            raise RuntimeError('%s has no matching target %s.*' %
                    (path, os.path.join(target_root, key)) )
        classname = os.path.basename(os.path.dirname(key))
        imagetriples.append((path, class_to_idx[classname], target_images[key]))
    return imagetriples

def make_parallel_dataset(image_roots, class_to_idx,
        intersection=False, verbose=None):
    """
    Returns (source, classnum, feature)
    """
    image_roots = [os.path.expanduser(d) for d in image_roots]
    image_sets = OrderedDict()
    for j, root in enumerate(image_roots):
        for path in walk_image_files(root, verbose=verbose):
            key = os.path.splitext(os.path.relpath(path, root))[0]
            if key not in image_sets:
                image_sets[key] = []
            if not intersection and len(image_sets[key]) != j:
                raise RuntimeError(
                    'Images not parallel: %s missing from one dir' % (key))
            image_sets[key].append(path)
    pairs = []
    for key, value in image_sets.items():
        if len(value) != len(image_roots):
            if intersection:
                continue
            else:
                raise RuntimeError(
                    'Images not parallel: %s missing from one dir' % (key))
        classname = os.path.basename(os.path.dirname(key))
        pairs.append((tuple(value), class_to_idx[classname]))
    return pairs

def make_class_dataset(source_root, class_to_idx, verbose=None):
    """
    Returns (source, classnum, feature)
    """
    imagepairs = []
    source_root = os.path.expanduser(source_root)
    for path in walk_image_files(source_root, verbose=verbose):
        classname = os.path.basename(os.path.dirname(path))
        imagepairs.append((path, class_to_idx[classname]))
    return imagepairs

