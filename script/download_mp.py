#!/usr/bin/env python2.7

# Script to create simple flat pytorch ImageFolder folder hierarchy
# of training and validation images for miniplaces.  Each category
# name is just a folder name (numbered in alphabetical order as in
# the original miniplaces), and both train and val images are places
# directly inside a single level of folders with the flat cateogry names.

import shutil, os, tarfile

def ensure_dir(dirname):
    try:
        os.makedirs(dirname)
    except:
        pass

# Download and untar data files.
gitdir = 'https://raw.githubusercontent.com/CSAILVision/miniplaces/master/data'
urls = [
    '%s/val.txt' % gitdir,
    '%s/train.txt' % gitdir,
    '%s/categories.txt' % gitdir,
    '%s/object_categories.txt' % gitdir,
    'http://miniplaces.csail.mit.edu/data/data.tar.gz',
]
ensure_dir('dataset/miniplaces/raw')
ensure_dir('dataset/miniplaces/data')

# python2 vs 3
try:
    import urllib
    urlopen = urllib.request.urlopen
except:
    import urllib2
    urlopen = urllib2.urlopen

for url in urls:
    filename = url.rpartition('/')[2]
    file_path = os.path.join('dataset/miniplaces/raw', filename)
    if not os.path.exists(file_path):
        print('Downloading %s' % url)
        data = urlopen(url)
        with open(file_path, 'wb') as f:
            f.write(data.read())
    if file_path.endswith('.tar.gz'):
        with tarfile.open(file_path) as tar:
            print('Untarring %s' % file_path)
            tar.extractall('dataset/miniplaces/data')
        # os.unlink(file_path)
    else:
        shutil.copyfile(file_path,
                os.path.join('dataset/miniplaces/data',
                    os.path.basename(file_path)))

# Now copy them into simple pytorch data format.
ensure_dir('dataset/miniplaces/simple/train')
ensure_dir('dataset/miniplaces/simple/val')

# Copy the train images to flat category directory names
categories = []
trainfiles = []
for root, dirs, files in os.walk("dataset/miniplaces/data/images/train"):
    files = [f for f in files if f.endswith('.jpg')]
    if not files:
        continue
    catname = '-'.join(root.split('/')[6:])
    categories.append(catname)
    ensure_dir('dataset/miniplaces/simple/train/%s' % catname)
    ensure_dir('dataset/miniplaces/simple/val/%s' % catname)
    print('Copying train/%s' % catname)
    for f in files:
        target = 'train/%s/%s' % (catname, f)
        trainfiles.append(target)
        shutil.copyfile(os.path.join(root, f),
                os.path.join('dataset/miniplaces/simple', target))

categories.sort()
# Save a file listing all images, which can be used to speed loading.
trainfiles.sort()
with open('dataset/miniplaces/simple/train.txt', 'w') as f:
    for filename in trainfiles:
        f.write('%s\n' % filename)

# Copy the val images to the same flat category directory names.
valfiles = []
with open('dataset/miniplaces/data/val.txt') as f:
    for line in f.readlines():
        fn, catnum = line.strip().split()
        basename = os.path.basename(fn)
        catname = categories[int(catnum)]
        target = 'val/%s/%s' % (catname, basename)
        print('Copying %s' % target)
        valfiles.append(target)
        shutil.copyfile(os.path.join('dataset/miniplaces/data/images', fn),
                 os.path.join('dataset/miniplaces/simple', target))

# Save a file listing all images, which can be used to speed loading.
valfiles.sort()
with open('dataset/miniplaces/simple/val.txt', 'w') as f:
    for filename in valfiles:
        f.write('%s\n' % filename)
