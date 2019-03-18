import pdb
import math
import torch
import random
import numpy as np
from itertools import product
from sklearn import metrics, mixture

def adjust_lr(iter, iter_per_epoch, config):
    """ Exponentially decaying learning rate
    Args:
        iter: current iteration
        iter_per_epoch: iterations per epoch
        config: configuation file
    Returns:
        decay_lr: decayed learning rate
    """
    if config["decay_every_epoch"] == -1:
        decay_lr = config["init_lr"]
    else:
        decay_lr = config["init_lr"] * math.exp(
            math.log(config["decay_factor"]) / iter_per_epoch / config["decay_every_epoch"])**iter
    return decay_lr

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim)-1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def to_data(ptdata, to_clone=True):
    # TODO: need to_clone?
    if to_clone:
        if type(ptdata) == type(list()):
            return [dt.detach().cpu().numpy() for dt in ptdata]
        else:
            return ptdata.detach().cpu().numpy()
    else:
        if type(ptdata) == type(list()):
            return [dt.cpu().numpy()for dt in ptdata]
        else:
            return ptdata.cpu().numpy()

def where(cond, x1, x2):
    """ Differentiable equivalent of np.where (or tf.where)
        Note that type of three variables should be same.
    Args:
        cond: condition
        x1: selected value if condition is 1 (True)
        x2: selected value if condition is 0 (False)
    """
    return (cond * x1) + ((1-cond) * x2)

def idx2onehot(idx, num_labels):
    """ Convert indices to onethot vector
    """
    B = idx.size(0)
    one_hot = idx.new_zeros((B, num_labels))
    one_hot.scatter_(1, idx.view(-1,1), 1)
    return one_hot

def get_location_feat_1D(loc_feat_dim, loc, length):
    loc_feat = torch.zeros(loc_feat_dim)
    new_sIdx = math.floor( loc[0] * (loc_feat_dim / length) )
    new_eIdx = math.floor( loc[1] * (loc_feat_dim / length) )
    loc_feat[new_sIdx:new_eIdx].fill_(1)
    return loc_feat

def make_list_at(values, idx):
    val_list = [v[idx] for v in values]
    return val_list

""" def transpose_list(list_data): return zip(*list_data) """
def transpose_stack_list(list_data, mode="default"):
    tr_data = zip(*list_data)
    if mode == "caption":
        return [torch.stack(item, dim=0).long()[:,:-1] for item in tr_data]
    elif mode == "caption_label":
        return [torch.stack(item, dim=0)[:,1:] for item in tr_data]
    else:
        return [torch.stack(item, dim=0) for item in tr_data]

""" Computation helpers """
def apply_on_sequence(layer, inp):
    inp = to_contiguous(inp)
    inp_size = list(inp.size())
    output = layer(inp.view(-1, inp_size[-1]))
    output = output.view(*inp_size[:-1], -1)
    return output

def compute_distance(inp1, inp2, method="l2", normalize=True):

    if inp1.dim() == 1:
        # distance between two vectors
        if normalize:
            inp1 = inp1 / inp1.norm(p=2, keepdim=True).expand_as(inp1)
            inp2 = inp2 / inp2.norm(p=2, keepdim=True).expand_as(inp2)
        if method == "l2":
            return torch.sqrt( ((inp1-inp2)**2).sum() ) # l2 distance
        elif method == "cosine":
            return (inp1 * inp2).sum() # cosine similarity
        else:
            raise NotImplementedError()
    elif inp1.dim() == 2:
        # distance between two batch of vectors
        if normalize:
            inp1 = inp1 / inp1.norm(p=2, dim=1, keepdim=True).expand_as(inp1)
            inp2 = inp2 / inp2.norm(p=2, dim=1, keepdim=True).expand_as(inp2)
        if method == "l2":
            return torch.sqrt( ((inp1-inp2)**2).sum(1) ) # l2 distance
        elif method == "cosine":
            return (inp1 * inp2).sum(1) # cosine similarity
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
