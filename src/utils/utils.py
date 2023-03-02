import torch
import numpy as np
import scipy.io
import h5py
import gdown
import glob
import os
from scipy.io import loadmat
import torch.nn as nn

import operator
from functools import reduce
from functools import partial

#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_finite_diff(c_norm, x, loc=0):
    """
    loc = 0, -1 for left, right, respectively
    returns: 1/c0 and c0-normalized differencing terms, 
            first/last term of c_norm is 1/c0, 1/cN for loc=0,-1. respectively.
    """
    len_c = len(c_norm)
    
    if loc==0:
        return c_norm[0], torch.sum(c_norm[1:] * x[...,1:len_c], dim=-1)
    elif loc==-1:
        return c_norm[-1], torch.sum(c_norm[:-1] * x[...,-len_c:-1], dim=-1)
    else:
        raise NotImplementedError
        

# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super().__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)

        return loss

# A simple feedforward neural network
class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c


class DataDownloader():
    dir_id = {
        "Heat_Neu_1D":"1xIe-lPFk7z91CeEZtSuD47H7ajxaJC2j",
        "Burgers_Per_1D":"1wr1rUpT4jSYJNEk81mK4elm5BzihIQ3b",
        "Burgers_Dir_1D":"1Ehxxj6751AzQChF6gh3TggCtWSPRLPOT",
        "Stokes_Dir_1D":"1ILpBKD__iddtm-CEp2j_UqjXkcM3vOXT",
        "Heat_Neu_2D":"1UwNzd40DiStP0GNn9VO0yytMDxQEc1y3",
        "Burgers_Per_2D":"1Wvnx-8_MJG9bUhrNnMQfOZsdkiro2x1v",
        "Burgers_Dir_2D":"1v4J5T2OAFgPOjEawqIyUqZVsISBZwfSg",
        "Stokes_Dir_2D":"1XdKe_4_TeEpoF3kYMDRC1-osDALg7D-N",
        "NV_Dir_3D":"125T9UvHIgmvabtxDy2a1hqXA1AwFREdv",
        "Wave_Neu_3D":"1JFlvlVpFAvRSx-RMbzTm6F2NvitNQtY5",
        }
    
    def __init__(
        self,
        output = 'Data',
        quiet = False,
        ):
        self.output = output
        self.quiet = quiet
        
    def download(self, id, tag=None):
        if id not in self.dir_id:
            assert 0, "Data ID not present in the repo. Check again! Supported ones are " \
                f"[{','.join(self.dir_id.keys())}]"
            
        if tag is None:
            gdown.download_folder(id=self.dir_id[id], output=os.path.join(self.output, id), quiet=self.quiet)
        else:
            id_and_names = self.dir_list(self.dir_id[id])
            located = False
            for child_id, child_name, child_type in id_and_names:
                if tag in child_name:
                    located=True
                    break
            if located:
                parent = os.path.join(self.output, id)
                if not os.path.exists(parent):
                    os.makedirs(parent)
                gdown.download(id=child_id, output=os.path.join(parent, child_name), quiet=self.quiet)
            else:
                assert 0, f"Provided tag:{tag} not present in the {id}"
                

    def locate(self, id, tag):
        files = glob.glob(os.path.join(self.output, id, '*.mat'))
        located = False
        for file in files:
            if tag in file:
                located = True
                break
        if located:
            return loadmat(file)
        else:
            assert 0, f"No file found with tag: {tag}."
            

    def dir_list(self, id):
        from gdown.download_folder import _parse_google_drive_file, _get_session
        
        url = "https://drive.google.com/drive/folders/{id}".format(id=id)
        sess = _get_session(proxy=None, use_cookies=True)
        # canonicalize the language into English

        if "?" in url:
            url += "&hl=en"
        else:
            url += "?hl=en"

        try:
            res = sess.get(url, verify=True)
        except requests.exceptions.ProxyError as e:
            print(
                "An error has occurred using proxy:", sess.proxies, file=sys.stderr
            )
            print(e, file=sys.stderr)
            return None

        if res.status_code != 200:
            return None

        gdrive_file, id_name_type_iter = _parse_google_drive_file(
            url=url,
            content=res.text,
        )
        return id_name_type_iter