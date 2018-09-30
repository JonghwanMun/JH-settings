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

class SSTSequenceEncoder(nn.Module):
    """
    Container module with 1D convolutions to generate proposals
    This code is from https://github.com/ranjaykrishna/SST/blob/master/models.py
    and modified for integration.
    """

    def __init__(self, config):
        super(SSTSequenceEncoder, self).__init__()

        # get options for SST
        self.video_dim = utils.get_value_from_dict(config, "sst_video_dim", 500)
        self.hidden_dim = utils.get_value_from_dict(config, "sst_hidden_dim", 512)
        self.K = utils.get_value_from_dict(config, "sst_K", 64) # number of proposals
        self.rnn_type = utils.get_value_from_dict(config, "sst_rnn_type", "GRU")
        self.rnn_num_layers = utils.get_value_from_dict(config, "sst_rnn_num_layer", 2)
        self.rnn_dropout = utils.get_value_from_dict(config, "sst_rnn_dropout", 0.2)

        # get layers of SST
        self.rnn = getattr(nn, self.rnn_type)(self.video_dim, self.hidden_dim,
                self.rnn_num_layers, batch_first=True, dropout=self.rnn_dropout)
        self.scores = torch.nn.Linear(self.hidden_dim, self.K)

    def forward(self, features):

        # dealing with batch size 1
        if len(features.size()) == 2:
            features = torch.unsqueeze(features, 0)
        B, L, _ = features.size()

        rnn_output, _ = self.rnn(features)
        rnn_output = rnn_output.contiguous()
        outputs = torch.sigmoid(self.scores(rnn_output.view(-1, self.hidden_dim)))
        outputs = outputs.view(B, L, self.K)
        outputs = torch.clamp(outputs, 0.001, 0.999) # TODO: decide to use this or not?
        return outputs, rnn_output

class Attention(nn.Module):
    def __init__(self, config, name=""):
        super(Attention, self).__init__()
        if name != "":
            name = name + "_"
        self.key_dim = utils.get_value_from_dict(config, name+"att_key_dim", 512)
        self.feat_dim = utils.get_value_from_dict(config, name+"att_feat_dim", 512)
        self.att_hid_dim = utils.get_value_from_dict(config, name+"att_hidden_dim", 512)

        self.key2att = nn.Linear(self.key_dim, self.att_hid_dim)
        self.feat2att = nn.Linear(self.feat_dim, self.att_hid_dim)
        self.to_alpha = nn.Linear(self.att_hid_dim, 1)

    def forward(self, key, feats, feat_masks=None):
        """ Compute attention weights and attended feature (weighted sum)
        Args:
            key: key vector to compute attention weights; [B, K]
            feats: features where attention weights are computed; [B, A, D]
            feat_masks: mask for effective features; [B, A]
        """
        # check inputs
        assert len(key.size()) == 2
        assert len(feats.size()) == 3 or len(feats.size()) == 4
        assert len(feat_masks.size()) == 2

        # dealing with dimension 4
        if len(feats.size()) == 4:
            is_spatial_feat = True
            B, W, H, D = feats.size()
            feats = feats.view(B, W*H, D)
        A = feats.size(1)

        # embedding key and feature vectors
        att_f = net_utils.apply_on_sequence(self.feat2att, feats)   # B * L * att_hid_dim
        att_k = self.key2att(key)                                   # B * att_hid_dim
        att_k = att_k.unsqueeze(1).expand_as(att_f)                 # B * L * att_hid_dim

        # compute attention weights
        dot = torch.tanh(att_f + att_k)                                 # B * L * att_hid_dim
        alpha = net_utils.apply_on_sequence(self.to_alpha, dot)     # B * L * 1
        alpha = alpha.view(-1, A)                                   # B * L
        weight = F.softmax(alpha, dim=1)                            # B * L

        if feat_masks is not None:
            weight = weight * feat_masks.float()                    # B * A
            weight = weight / (weight.sum(1, keepdim=True) + 1e-8)  # re-normalize

        # compute weighted sum: bmm working on (B, 1, A) * (B, A, D) -> (B, 1, D)
        att_feats = torch.bmm(weight.unsqueeze(1), feats).squeeze(1) # B * D
        return att_feats

class TemporalDynamicAttention(nn.Module):
    def __init__(self, config, name=""):
        super(TemporalDynamicAttention, self).__init__()
        if name != "":
            name = name + "_"
        self.ctx_dim = utils.get_value_from_dict(config, name+"att_ctx_dim", 512)
        self.prevh_dim = utils.get_value_from_dict(config, name+"att_prevh_dim", 512)
        self.feat_dim = utils.get_value_from_dict(config, name+"att_feat_dim", 512)
        self.att_hid_dim = utils.get_value_from_dict(config, name+"att_hidden_dim", 512)

        self.ctx2att = nn.Linear(self.ctx_dim, self.att_hid_dim)
        self.prevh2att = nn.Linear(self.prevh_dim, self.att_hid_dim)
        self.feat2att = nn.Linear(self.feat_dim, self.att_hid_dim)
        self.to_alpha = nn.Linear(self.att_hid_dim, 1)

    def forward(self, prevh, ctx, feats, feat_masks=None):
        """ Compute attention weights and attended feature (weighted sum)
        Args:
            prevh: previous hidden states to compute attention weights; [B, K]
            ctx: context vectors for eash proposals from Bi-directional SST; [B, K]
            feats: features where attention weights are computed; [B, A, D]
            feat_masks: mask for effective features; [B, A]
        """
        # check inputs
        assert len(prevh.size()) == 2
        assert len(feats.size()) == 3 or len(feats.size()) == 4
        assert len(feat_masks.size()) == 2

        # dealing with dimension 4
        if len(feats.size()) == 4:
            is_spatial_feat = True
            B, W, H, D = feats.size()
            feats = feats.view(B, W*H, D)
        A = feats.size(1)

        # embedding prevh, context and feature vectors
        att_f = net_utils.apply_on_sequence(self.feat2att, feats)   # B * L * att_hid_dim
        att_h = self.prevh2att(prevh)                               # B * att_hid_dim
        att_h = att_h.unsqueeze(1).expand_as(att_f)                 # B * L * att_hid_dim
        att_c = self.ctx2att(ctx)                                   # B * att_hid_dim
        att_c = att_c.unsqueeze(1).expand_as(att_f)                 # B * L * att_hid_dim

        # compute attention weights
        dot = torch.tanh(att_f + att_h + att_c)                         # B * L * att_hid_dim
        alpha = net_utils.apply_on_sequence(self.to_alpha, dot)     # B * L * 1
        alpha = alpha.view(-1, A)                                   # B * L
        weight = F.softmax(alpha, dim=1)                            # B * L

        if feat_masks is not None:
            weight = weight * feat_masks.float()                    # B * A
            weight = weight / (weight.sum(1, keepdim=True) + 1e-8)  # re-normalize to 1

        # compute weighted sum: bmm working on (B, 1, A) * (B, A, D) -> (B, 1, D)
        att_feats = torch.bmm(weight.unsqueeze(1), feats).squeeze(1) # B * D
        # concatenate with context vectors
        #att_feats = torch.cat([att_feats, ctx],dim=1)
        return att_feats

class ContextGating(nn.Module):
    def __init__(self, config, name=""):
        super(ContextGating, self).__init__()
        if name != "":
            name = name + "_"

        # TODO: gc -> cg
        cg_emb_dim = config.get(name + "cg_emb_dim", 512)
        att_dim = config.get(name + "cg_att_feat_dim", 512)
        ctx_dim = config.get(name + "cg_ctx_feat_dim", 512)
        cg_key_dim = config.get(name + "cg_key_dim", 512)
        self.prevh_dim = utils.get_value_from_dict(config, name+"att_prevh_dim", 512)

        self.W_att = nn.Linear(att_dim, cg_emb_dim)
        self.W_ctx = nn.Linear(ctx_dim, cg_emb_dim)
        self.W_g = nn.Linear(cg_key_dim, cg_emb_dim)

    def forward(self, att_feat, ctx_feat, xt, prevh):
        """ Compute attention weights and attended feature (weighted sum)
        Args:
            att_feat: attended features; [B, A, D]
            ctx_feat: context vectors for eash proposals from Bi-directional SST; [B, K]
            xt: word embedding
            prevh: previous hidden states to compute attention weights; [B, K]
        """
        if prevh is None:
            prevh = att_feat.new_zeros((att_feat.size(0), self.prevh_dim))

        emb_att = torch.tanh(self.W_att(att_feat))
        emb_ctx = torch.tanh(self.W_ctx(ctx_feat))
        g_ctx = torch.sigmoid(self.W_g( torch.cat([emb_att, emb_ctx, xt, prevh], dim=1) ))
        gated_feat = torch.cat([(1-g_ctx) * emb_att, g_ctx * emb_ctx], dim=1)

        return gated_feat


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

class WeightedBCE(nn.Module):
    """
    Weighted BCE is from https://github.com/ranjaykrishna/SST/blob/master/models.py
    and modified for integration.
    """

    def __init__(self, prefix="", w=None):
        super(WeightedBCE, self).__init__()
        self.w = w
        self.prefix = prefix
        if prefix is not "":
            self.prefix = prefix + "_"

    def set_w(self, w):
        self.w = w

    def get_w(self):
        return self.w

    def forward(self, outputs, gts):
        outputs = outputs[self.prefix + "logits"] # [B,L,K]
        labels = gts[self.prefix + "labels"].type_as(outputs) # [B,L,K]

        w = torch.FloatTensor(self.w).type_as(outputs) #[K]
        term0 = w * labels
        term1 = (1. - w) * (1. - labels)

        loss = - ((term0.view(-1) * outputs.log().view(-1)) \
                 + (term1.view(-1) * (1. - outputs).log().view(-1)))
        return loss.mean()
        #return loss.sum() / (outputs.size(0) * outputs.size(1))

class SequenceGeneratorCriterion(nn.Module):
    """ Captioning (or Language model) criterion
    Negative log likelihood over all time steps
    """
    def __init__(self, is_mft=False, use_bce=False):
        super(SequenceGeneratorCriterion, self).__init__()
        self.is_mft = is_mft
        self.use_bce = use_bce

    def forward(self, outputs, gts):
        logits = outputs["seq_gen_logits"]
        targets = outputs["seq_gen_target_labels"]
        if isinstance(logits, list) and isinstance(targets, list):
            assert len(logits) == len(targets), "{} != {}".format(len(logits), len(targets))

        loss = 0
        num_step = len(logits)
        for step in range(num_step):
            if self.is_mft:
                step_loss = F.binary_cross_entropy(logits[:,step], targets[:,step])
            else:
                if self.use_bce:
                    step_loss = F.binary_cross_entropy_with_logits(logits[step], targets[step])
                else:
                    step_loss = F.cross_entropy(logits[step], targets[step])
            loss += step_loss
        return loss / num_step

    def set_itow(self, itow):
        self.itow = itow

class CaptioningCriterion(nn.Module):
    """ Captioning (or Language model) criterion
    Negative log likelihood over all time steps
    """
    def __init__(self):
        super(CaptioningCriterion, self).__init__()

    def forward(self, outputs, gts):
        logprobs = outputs["caption_logprobs"]
        target = outputs["caption_labels"]
        mask = (target > 0).float()

        assert len(logprobs.size()) == 3    # [B, L, W]
        assert len(target.size()) == 2      # [B, L]

        loss = -logprobs.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        loss = torch.sum(loss) / torch.sum(mask)
        return loss

    def set_itow(self, itow):
        self.itow = itow

class HierarchicalCaptioningCriterion(nn.Module):
    """ Captioning (or Language model) criterion
    Negative log likelihood over all time steps
    """
    def __init__(self):
        super(HierarchicalCaptioningCriterion, self).__init__()

    def forward(self, outputs, gts):
        seq_logprobs = outputs["caption_logprobs"]
        seq_target = outputs["caption_labels"]
        seq_masks = outputs["seq_masks"]

        total_loss, num_seq = 0, 0
        for logprobs,target,seq_mask in zip(seq_logprobs, seq_target, seq_masks):
            assert len(logprobs.size()) == 3    # [B, L, W]
            assert len(target.size()) == 2      # [B, L]
            num_seq += seq_mask.sum().item()

            word_mask = (target > 0).float()
            if word_mask.sum().item() == 0:
                continue
            selected_logprobs = -logprobs.gather(2, target.unsqueeze(2)).squeeze(2)
            #selected_logprobs[word_mask == 0] = 0 # TODO: need this?
            loss = selected_logprobs * word_mask
            total_loss += (torch.sum(loss) / (torch.sum(word_mask) + 1e-8))
        return total_loss / num_seq

    def set_itow(self, itow):
        self.itow = itow

class CaptioningRewardCriterion(nn.Module):
    """ Captioning (or Language model) criterion with reward (RL)
    Negative log likelihood over all time steps weighted by rewards
    """
    def __init__(self, config):
        super(CaptioningRewardCriterion, self).__init__()
        self.use_all_gt = config["model"].get("use_all_gt", False)
        self.use_tiou_reward = config["model"].get("use_tiou_reward", False)
        self.tiou_baseline = config["model"].get("tiou_baseline", 0.5)
        self.use_caption_reward = config["model"].get("use_caption_reward", True)
        assert (self.use_tiou_reward is True) or (self.use_caption_reward is True), \
                "One of reward [tiou|caption] should be used"

        if self.use_caption_reward:
            self.caption_rewarder= Rewards(config)

    def forward(self, outputs, gts):
        sampled_logprobs = outputs["sampled_logprobs"]
        matched_gt_sequences = outputs["matched_gt_sequences"]
        seq_masks = outputs["seq_masks"]
        caption_masks = outputs["caption_masks"]
        if self.use_tiou_reward:
            matched_gt_sequence_tious = outputs["matched_gt_sequence_tious"]
        all_gt = None

        if self.use_caption_reward:
            model_captions = self.convert_to_caption(
                    outputs["sampled_labels"], seq_masks)
            baseline_captions = self.convert_to_caption(
                    outputs["baseline_labels"], seq_masks)
            gt_captions = self.convert_gt_caption(matched_gt_sequences, seq_masks)
            if self.use_all_gt:
                all_gt = [utils.label2string(self.itow, cc, 3) for cc in outputs["all_gt_sequences"]]
                all_gt = " ".join(all_gt)

            model = self.caption_rewarder.compute_caption_scores(model_captions, gt_captions, all_gt)
            baseline = self.caption_rewarder.compute_caption_scores(baseline_captions, gt_captions, all_gt)
            caption_rewards = [m - b for m,b in zip(model,baseline)]

        loss, num_pp = 0, 0
        cap_num = len(sampled_logprobs)
        seq_num, _, _ = sampled_logprobs[0].size()
        for ci in range(cap_num):
            for si in range(seq_num):
                if seq_masks[si,ci].item() == 1:
                    num_preds = sampled_logprobs[ci][si].size(0)
                    reward = 0
                    if self.use_caption_reward:
                        reward += caption_rewards[si][ci]
                    if self.use_tiou_reward:
                        reward += (matched_gt_sequence_tious[si][ci] - self.tiou_baseline)
                    loss = loss + (-sampled_logprobs[ci][si][:, 0]
                            * caption_masks[ci][si][:num_preds]
                            * reward).sum()
                    num_pp += 1

        return loss / num_pp

    def set_itow(self, itow):
        self.itow = itow

    def convert_to_caption(self, idxs, seq_masks, concat=False):
        """ Convert idxs to sentences
        Args:
            idxs: idxs from event sequence; seqLen*[seqNum,capLen]
            seq_masks: mask for event sequence; [seqNum,seqLen]
        """
        seq_len = len(idxs)
        seq_num, cap_len = idxs[0].size()
        output = [list() for n in range(seq_num)]
        for seq_step,step_idxs in enumerate(idxs):
            for b in range(seq_num):
                if seq_masks[b,seq_step].item() == 1:
                    cur_caption = utils.label2string(self.itow, step_idxs[b], end_idx=3)
                    output[b].append(cur_caption)

        if concat:
            output = [" ".join(out) for out in output]

        return output

    def convert_gt_caption(self, idxs, seq_masks, concat=False):
        """ Convert idxs to sentences
        Args:
            idxs: idxs from event sequence; seqLen*[seqNum,capLen]
            seq_masks: mask for event sequence; [seqNum,seqLen]
        """
        output = []
        for si,seq_idxs in enumerate(idxs):
            gt_seq = []
            for ci,cap_idxs in enumerate(seq_idxs):
                if seq_masks[si,ci].item() == 1:
                    cur_caption = utils.label2string(self.itow, cap_idxs, end_idx=3)
                    gt_seq.append(cur_caption)
            output.append(gt_seq)

        if concat:
            output = [" ".join(out) for out in output]

        return output
