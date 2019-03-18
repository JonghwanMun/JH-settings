import pdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import utils, io_utils, net_utils

"""
Blocks for layers
e.g., conv1d, conv2d, linear, mlp, etc
"""
def get_conv1d(in_dim, out_dim, k_size, stride=1, padding=0, bias=True,
               dropout=0.0, nonlinear="ReLU", use_batchnorm=False):
    layers = []
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(in_dim))
    layers.append(nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=k_size, \
                            stride=stride, padding=padding, bias=bias))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(out_dim))
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    if nonlinear != "None":
        layers.append(getattr(nn, nonlinear)())
    return nn.Sequential(*layers)

def get_conv2d(in_dim, out_dim, k_size, stride=1, padding=0, bias=True,
               dropout=0.0, nonlinear="ReLU", use_batchnorm=False):
    layers = []
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(in_dim))
    layers.append(nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=k_size, \
                            stride=stride, padding=padding, bias=bias))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(out_dim))
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    if nonlinear != "None":
        layers.append(getattr(nn, nonlinear)())
    return nn.Sequential(*layers)

def get_linear(in_dim, out_dim, bias=True, dropout=0.0, nonlinear="ReLU", use_batchnorm=False):
    layers = []
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(in_dim))
    layers.append(nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(out_dim))
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    if nonlinear != "None":
        layers.append(getattr(nn, nonlinear)())
    return nn.Sequential(*layers)

def get_mlp(in_dim, out_dim, hidden_dims, bias=True, dropout=0.0, nonlinear="ReLU", use_batchnorm=False):
    layers = []
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(in_dim))

    D = in_dim
    for dim in hidden_dims:
        layers.append(nn.Linear(in_features=D, out_features=dim, bias=bias))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(dim))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        if nonlinear != "None":
            layers.append(getattr(nn, nonlinear)())
        D = dim
    layers.append(nn.Linear(D, out_dim))
    return nn.Sequential(*layers)

def get_mlp2d(in_dim, out_dim, hidden_dims, bias=True, dropout=0.0,
              nonlinear="ReLU", use_batchnorm=False):
    layers = []
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(in_dim))

    D = in_dim
    for dim in hidden_dims:
        layers.append(nn.Linear(in_features=D, out_features=dim, bias=bias))
        layers.append(nn.Conv2d(in_channels=D, out_channels=dim, kernel_size=k_size, \
                            stride=stride, padding=padding, bias=bias))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(dim))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        if nonlinear != "None":
            layers.append(getattr(nn, nonlinear)())
        D = dim
    layers.append(nn.Conv2d(D, out_dim, k_size, stride, padding, bias=bias))
    return nn.Sequential(*layers)

def get_res_block_2d(in_dim, out_dim, hidden_dim):
    layers = []
    # 1st conv
    layers.append(nn.Conv2d(in_dim, hidden_dim, 1, 1, bias=False))
    layers.append(nn.BatchNorm2d(hidden_dim))
    layers.append(nn.ReLU(inplace=True))
    # 2nd conv
    layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False))
    layers.append(nn.BatchNorm2d(hidden_dim))
    layers.append(nn.ReLU(inplace=True))
    # 3rd conv
    layers.append(nn.Conv2d(hidden_dim, out_dim, 1, 1, bias=False))
    layers.append(nn.BatchNorm2d(out_dim))

    return nn.Sequential(*layers)

def get_embedding(num_embeddings, emb_dim, padding_idx=0,
                  dropout=0.0, nonlinear="ReLU"):
    layers = []
    layers.append(nn.Embedding(num_embeddings, emb_dim))
    if nonlinear != "None":
        layers.append(getattr(nn, nonlinear)())
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

"""
Layers for networks
"""
class WordEmbedding(nn.Module):
    def __init__(self, config, name=""):
        super(WordEmbedding, self).__init__() # Must call super __init__()
        if name != "":
            name = name + "_"

        # get configuration
        inp_dim = utils.get_value_from_dict(
            config, name+"word_emb_vocab_size", 256)
        out_dim = utils.get_value_from_dict(
            config, name+"word_emb_dim", 52)
        dropout_prob = utils.get_value_from_dict(
            config, name+"word_emb_dropout_prob", 0)
        nonlinear = utils.get_value_from_dict(
            config, name+"word_emb_nonlinear_fn", "ReLU")

        # set layers
        self.word_emb = get_embedding(inp_dim, out_dim,
                dropout=dropout_prob, nonlinear=nonlinear)

    def forward(self, inp):
        """
        Args:
            inp: [B, L]; one-hot vector
        Returns:
            answer_label : [B, L, out_dim]
        """
        return self.word_emb(inp)

class MLP(nn.Module):
    def __init__(self, config, name=""):
        super(MLP, self).__init__() # Must call super __init__()
        if name != "":
            name = name + "_"

        # get configuration
        inp_dim = utils.get_value_from_dict(
            config, name+"mlp_inp_dim", 256)
        out_dim = utils.get_value_from_dict(
            config, name+"mlp_out_dim", 52)
        dropout_prob = utils.get_value_from_dict(
            config, name+"mlp_dropout_prob", 0)
        hidden_dim = utils.get_value_from_dict(
            config, name+"mlp_hidden_dim", (1024,))
        use_batchnorm = utils.get_value_from_dict(
            config, name+"mlp_use_batchnorm", False)
        nonlinear = utils.get_value_from_dict(
            config, name+"mlp_nonlinear_fn", "ReLU")

        # set layers
        self.mlp_1d = get_mlp(inp_dim, out_dim, hidden_dim, \
                dropout=dropout_prob, nonlinear=nonlinear, use_batchnorm=use_batchnorm)

    def forward(self, inp):
        """
        Args:
            inp: [B, inp_dim]
        Returns:
            answer_label : [B, out_dim]
        """
        return self.mlp_1d(inp)

    def Inference_forward(self, inp):
        output = []
        for name, module in self.mlp_1d._modules.items():
            inp = module(inp)
            output.append(inp)
        return output

class ResBlock2D(nn.Module):
    def __init__(self, config, name=""):
        super(ResBlock2D, self).__init__() # Must call super __init__()
        if name != "":
            name = name + "_"

        # get configuration
        inp_dim = utils.get_value_from_dict(
            config, name+"res_block_2d_inp_dim", 1024)
        out_dim = utils.get_value_from_dict(
            config, name+"res_block_2d_out_dim", 1024)
        hidden_dim = utils.get_value_from_dict(
            config, name+"res_block_2d_hidden_dim", 512)
        self.num_blocks = utils.get_value_from_dict(
            config, name+"num_blocks", 1)
        self.use_downsample = utils.get_value_from_dict(
            config, name+"use_downsample", False)
        self.use_attention_transfer = utils.get_value_from_dict(
            config, "use_attention_transfer", False)

        # set layers
        if self.use_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(inp_dim, out_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_dim),
            )
        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            self.blocks.append(get_res_block_2d(inp_dim, out_dim, hidden_dim))
            if (i == 0) and self.use_downsample:
                inp_dim = out_dim

    def forward(self, inp):
        """
        Args:
            inp: [B, inp_dim, H, w]
        Returns:
            answer_label : [B, out_dim, H, w]
        """
        if self.use_attention_transfer:
            att_groups = []
        residual = inp
        for i in range(self.num_blocks):
            out = self.blocks[i](residual)
            if (i == 0) and self.use_downsample:
                residual = self.downsample(residual)
            out += residual
            out = F.relu(out)
            residual = out
            if self.use_attention_transfer:
                att_groups.append(out)

        return_val = {
            "feats": out,
        }
        if self.use_attention_transfer:
            return_val["att_groups"] = att_groups
        return return_val

class Embedding2D(nn.Module):
    def __init__(self, config, name=""):
        super(Embedding2D, self).__init__() # Must call super __init__()
        if name != "":
            name = name + "_"

        inp_dim = utils.get_value_from_dict(
            config, name+"emb2d_inp_dim", 1024)
        out_dim = utils.get_value_from_dict(
            config, name+"emb2d_out_dim", 256)
        dropout_prob = utils.get_value_from_dict(
            config, name+"emb2d_dropout_prob", 0.0)
        nonlinear = utils.get_value_from_dict(
            config, name+"emb2d_nonlinear_fn", "None")
        batchnorm = utils.get_value_from_dict(
            config, name+"emb2d_use_batchnorm", False)
        self.apply_l2_norm = \
            utils.get_value_from_dict(config, name+"emb2d_apply_l2_norm", False)
        self.only_l2_norm = utils.get_value_from_dict(
            config, name+"emb2d_only_l2_norm", False)

        assert not ((self.apply_l2_norm == False) and (self.only_l2_norm == True)), \
            "You set only_l2_norm as True, but also set apply_l2_norm as False"

        # define layers
        if not self.only_l2_norm:
            self.embedding_2d = get_conv2d(inp_dim, out_dim, 1, 1,
                dropout=dropout_prob, nonlinear=nonlinear, use_batchnorm=batchnorm)

    def forward(self, inp):
        """
        Args:
            inp: [batch_size, inp_dim, h, w]
        Returns:
            x: [batch_size, out_dim, h, w]
        """
        if self.apply_l2_norm:
            inp_size = inp.size()
            inp = inp.transpose(1,2).transpose(2,3)
            inp = inp.resize(inp_size[0], inp_size[2]*inp_size[3], inp_size[1])
            inp = F.normalize(inp, p=2, dim=1)
            inp = inp.resize(inp_size[0], inp_size[2], inp_size[3], inp_size[1])
            inp = inp.transpose(3,2).transpose(2,1)
            if self.only_l2_norm:
                return inp

        out = self.embedding_2d(inp)
        return out


""" Criterions """
class MultipleCriterions(nn.Module):
    """ Container for multiple criterions.
    Since pytorch does not support ModuleDict(), we use ModuleList() to
    maintain multiple criterions.
    """
    def __init__(self, names=None, modules=None):
        super(MultipleCriterions, self).__init__()
        if names is not None:
            assert len(names) == len(modules)
        self.names = names if names is not None else []
        self.crits = nn.ModuleList(modules) if modules is not None else nn.ModuleList()
        self.name2crit = {}
        if names is not None:
            self.name2crit = {name:self.crits[i]for i,name in enumerate(names)}

    def forward(self, crit_inp, gts):
        self.loss = {}
        self.loss["total_loss"] = 0
        for name,crit in self.get_items():
            self.loss[name] = crit(crit_inp, gts)
            self.loss["total_loss"] += self.loss[name]
        return self.loss

    def add(self, name, crit):
        self.names.append(name)
        self.crits.append(crit)
        self.name2crit[name] = self.crits[-1]

    def get_items(self):
        return iter(zip(self.names, self.crits))

    def get_names(self):
        return self.names

    def get_crit_by_name(self, name):
        return self.name2crit[name]

