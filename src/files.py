'''
Utilities for saving and loading torch tensors to .npy and .npz files.

Handles various cases such as torch sparse arrays and scalar data.
'''

import os
import numpy
import torch
from numpy import load as numpy_load
from numpy.lib.format import open_memmap

created_dirs = set()
def ensure_dir_for(filename):
    dirname = os.path.dirname(filename)
    if dirname in created_dirs:
        return
    try:
        created_dirs.add(dirname)
        os.makedirs(dirname)
    except:
        pass

def save_torch_npy(directory, filename, data):
    filename = os.path.join(directory, filename)
    ensure_dir_for(filename)
    numpydata = {}
    if hasattr(data, 'cpu'):
        data = data.cpu()
    if hasattr(data, 'is_sparse') and data.is_sparse:
        data = data.to_dense()
    if hasattr(data, 'numpy'):
        data = data.numpy()
    numpy.save(filename, data)

def load_npy_memmap(directory, filename):
    filename = os.path.join(directory, filename)
    return numpy.load(filename, mmap_mode='r')

def create_npy_memmap(directory, filename, shape=None, dtype='float32'):
    filename = os.path.join(directory, filename + '-inc')
    ensure_dir_for(filename)
    return open_memmap(filename, mode='w+', dtype=dtype, shape=shape)

def finish_npy_memmap(array, delete=False):
    '''
    When a new mmap is created, it starts with the name filename-inc.
    Call this finish function on the array to flush it and
    rename it without the inc.
    '''
    if isinstance(array, numpy.memmap) and array.mode and (
            'w' in array.mode or 'r+' in array.mode):
        # Note: for some reason, even though we open with mode w+, the
        # mmap object reports mode 'r+' afterwards.  No worries.
        if delete:
            os.remove(array.filename)
        else:
            array.flush()
            # Renaming from '-inc' is the way to make an atomic commit,
            # when data is finished computing.
            if array.filename.endswith('-inc'):
                fixedname = array.filename[:-4]
                os.rename(array.filename, fixedname)
                array.filename = fixedname
        if delete:
            del array

def save_torch_zip(directory, filename, data):
    filename = os.path.join(directory, filename)
    ensure_dir_for(filename)
    numpydata = {}
    for k, v in data.items():
        if hasattr(v, 'cpu'):
            v = v.cpu()
        if hasattr(v, 'is_sparse') and v.is_sparse:
            numpydata['%s indices' % k] = v._indices().numpy()
            numpydata['%s values' % k] = v._values().numpy()
            numpydata['%s shape' % k] = v.shape
        elif hasattr(v, 'numpy'):
            numpydata[k] = v.numpy()
        else:
            numpydata[k] = v
    numpy.savez(filename, **numpydata)

def load_torch_zip(directory, filename, varnames, cuda=False, numpy=False):
    filename = os.path.join(directory, filename)
    loaded = numpy_load(filename)
    result = []
    for k in varnames:
        if k in loaded:
            val = loaded[k]
            if len(val.shape) == 0:
                # Load scalars as scalars
                result.append(val[()])
            else:
                result.append(torch.from_numpy(loaded[k]))
        elif ('%s indices' % k) in loaded:
            values = torch.from_numpy(loaded['%s values' % k])
            indices = torch.from_numpy(loaded['%s indices' % k])
            shape = loaded['%s shape' % k]
            i_type, SparseTensor = sparse_types_for_dense_type(type(values))
            result.append(SparseTensor(indices, values, torch.Size(shape)))
        else:
            result.append(None)
    if cuda:
        result = [r.cuda() if hasattr(r, 'cuda') else r for r in result]
    elif numpy:
        result = [r.numpy() if hasattr(r, 'numpy') else r for r in result]
    return result

def path_split_all(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

def sparse_types_for_dense_type(dense_type):
    if 'cuda' in dense_type: # dense_type.is_cuda:
        index_type = torch.cuda.LongTensor
        if dense_type == 'torch.cuda.FloatTensor':
            sparse_type = torch.cuda.sparse.FloatTensor
        elif dense_type == 'torch.cuda.DoubleTensor':
            sparse_type = torch.cuda.sparse.DoubleTensor
        elif dense_type == 'torch.cuda.HalfTensor':
            sparse_type = torch.cuda.sparse.HalfTensor
        elif dense_type == 'torch.cuda.ByteTensor':
            sparse_type = torch.cuda.sparse.ByteTensor
        elif dense_type == 'torch.cuda.CharTensor':
            sparse_type = torch.cuda.sparse.CharTensor
        elif dense_type == 'torch.cuda.ShortTensor':
            sparse_type = torch.cuda.sparse.ShortTensor
        elif dense_type == 'torch.cuda.IntTensor':
            sparse_type = torch.cuda.sparse.IntTensor
        elif dense_type == 'torch.cuda.LongTensor':
            sparse_type = torch.cuda.sparse.LongTensor
        else:
            assert False, dense_type
    else:
        index_type = torch.LongTensor
        if dense_type == 'torch.FloatTensor':
            sparse_type = torch.sparse.FloatTensor
        elif dense_type == 'torch.DoubleTensor':
            sparse_type = torch.sparse.DoubleTensor
        elif dense_type == 'torch.HalfTensor':
            sparse_type = torch.sparse.HalfTensor
        elif dense_type == 'torch.ByteTensor':
            sparse_type = torch.sparse.ByteTensor
        elif dense_type == 'torch.CharTensor':
            sparse_type = torch.sparse.CharTensor
        elif dense_type == 'torch.ShortTensor':
            sparse_type = torch.sparse.ShortTensor
        elif dense_type == 'torch.IntTensor':
            sparse_type = torch.sparse.IntTensor
        elif dense_type == 'torch.LongTensor':
            sparse_type = torch.sparse.LongTensor
        else:
            assert False, dense_type
    return index_type, sparse_type

if __name__ == '__main__':
    from numpy.testing import assert_equal
    import shutil
    # Test saving and loading a variety of data types and shapes.
    a, b, c, d = (torch.randn(*shape, out=torch.DoubleTensor())
            for shape in [(1,), (2,3), (5,3,2,4), (2, 20)])
    b = b.float()
    c = c.half()
    nz = (d > 0.9).nonzero()
    d = torch.sparse.DoubleTensor(nz.t(), d[nz[:,0],nz[:,1]])
    s = 7
    save_torch_zip('testdir', 'test.npz', dict(a=a, b=b, c=c, d=d, s=s))
    at, bt, ct, dt, st = load_torch_zip('testdir', 'test.npz',
            ['a','b','c','d', 's'])
    assert_equal(a.numpy(), at.numpy())
    assert_equal(b.numpy(), bt.numpy())
    assert_equal(c.numpy(), ct.numpy())
    assert_equal(d._indices().numpy(), dt._indices().numpy())
    assert_equal(d._values().numpy(), dt._values().numpy())
    assert_equal(d.to_dense().numpy(), dt.to_dense().numpy())
    assert_equal(s, st)
    [bn] = load_torch_zip('testdir', 'test.npz', ['b'], numpy=True)
    assert_equal(b.numpy(), bn)
    shutil.rmtree('testdir')
