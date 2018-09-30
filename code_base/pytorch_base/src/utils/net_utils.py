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
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
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
"""
def apply_on_sequence(layer, inp):
    inp = to_contiguous(inp)
    inp_size = list(inp.size())
    output = layer(inp.view(-1, inp_size[-1]))
    new_size = inp_size[:-1]; new_size.append(output.shape[-1])
    output = output.view(new_size)
    return output
"""

def compute_distance(inp1, inp2, method="l2", normalize=True):
    if inp1.dim() == 1:
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




def nms_detections(props, scores, overlap=0.7, captions=None, th=0.0):

        props = props[scores > th]
        if captions:
            captions = captions[scores > th]

        scores = scores[scores > th]
        t1 = props[:, 0]
        t2 = props[:, 1]
        ind = np.argsort(scores)
        area = (t2 - t1 + 1).astype(float)
        pick = []
        while len(ind) > 0:
            i = ind[-1]
            pick.append(i)
            ind = ind[:-1]
            tt1 = np.maximum(t1[i], t1[ind])
            tt2 = np.minimum(t2[i], t2[ind])
            wh = np.maximum(0., tt2 - tt1 + 1.0)
            o = wh / (area[i] + area[ind] - wh)
            ind = ind[np.nonzero(o <= overlap)[0]]

        if captions:
            nms_props, nms_scores, nms_caps = props[pick, :], scores[pick], captions[pick]
            return nms_props, nms_scores, nms_caps
        else:
            nms_props, nms_scores = props[pick, :], scores[pick]
            return nms_props, nms_scores

def nms_all_predictions(preds, overlap=0.7, th=0.3):
    new_preds = {}
    for vid,pred in preds.items():
        pps, scs, new_preds[vid] = [], [], []
        for p in pred:
            pps.append(np.array(p["timestamp"]))
            scs.append(np.array(p["eventness"]))
            new_pp, new_sc = nms_detections(np.asarray(pps), np.asarray(scs), overlap)

        for npp,nsc in zip(new_pp,new_sc):
            new_preds[vid].append({"timestamp": npp, "eventness": nsc})
    return new_preds

def get_iou(pp1, pp2):
    intersection = max(0, min(pp1[1], pp2[1]) - max(pp1[0], pp2[0]))
    union = min(max(pp1[1], pp2[1]) - min(pp1[0], pp2[0]), pp1[1]-pp1[0] + pp2[1]-pp2[0])
    return float(intersection) / (union + 1e-8)

def is_pp_in_path(path, pp_idx):
    is_in = False
    for elem in path:
        flag, idx = elem.split("-")
        if int(idx) == pp_idx:
            return True
    return is_in

def get_iou_to_path(overlaps, path, pp_idx):
    ious = []
    for elem in path:
        flag, idx = elem.split("-")
        if flag == "GC":
            cur_iou = overlaps[int(idx), pp_idx]
            ious.append(cur_iou)

    if len(ious) == 0:
        return 0.0
    return max(ious)

def get_best_pp_from_path(path):
    pp_idx = []
    for elem in path:
        flag, idx = elem.split("-")
        if flag == "GC":
            pp_idx.append(int(idx))

    return pp_idx


def apply_dp(cand_pps, cand_scs, w=0.5, verbose=False, debug=False):
    n_step = len(cand_pps)

    # initialize score map (at step 1)
    dp_map = np.zeros((n_step, 2, n_step))
    dp_map[0,0,:] = np.asarray(cand_scs).squeeze()

    # initialize # overlap # ratios
    overlaps = np.zeros((n_step, n_step))
    for ns1 in range(n_step):
        for ns2 in range(n_step):
            if ns1 < ns2: continue
            if ns1 == ns2: overlaps[ns1,ns2] = 1

            ratio = get_iou(cand_pps[ns1], cand_pps[ns2])
            overlaps[ns1,ns2] = ratio
            overlaps[ns2,ns1] = ratio

    paths = np.empty((n_step,2,n_step), dtype=object)
    paths.fill([])
    for ns in range(n_step):
        paths[0,0,ns] = ["GC-{}".format(ns)]
        paths[0,1,ns] = ["NC-{}".format(ns)]

    for step in range(1,n_step):
        prev_paths = paths[step-1]

        for cur_c,cur_p in product(range(2), range(n_step)):
            tmp_score = np.zeros((2, n_step))

            for prev_c,prev_p in product(range(2), range(n_step)):
                # if proposal p already exists in path of previous time step we should exclude at this time step
                if is_pp_in_path(paths[step-1,prev_c,prev_p], cur_p):
                    # if proposal p already exists in path of previous time step we should exclude at this time step
                    tmp_score[prev_c,prev_p] = -9999
                else:
                    max_iou = get_iou_to_path(overlaps, paths[step-1,prev_c,prev_p], cur_p)
                    if cur_c == 0:
                        tmp_score[prev_c,prev_p] = (1 - max_iou) + w * cand_scs[cur_p]
                    else:
                        tmp_score[prev_c,prev_p] = max_iou + w * (1 - cand_scs[cur_p])

            tmp_score = tmp_score + dp_map[step-1]
            ind = np.unravel_index(np.argmax(tmp_score, axis=None), tmp_score.shape)
            if cur_c == 0:
                cur_path = [*paths[step-1,ind[0],ind[1]], "GC-{}".format(cur_p)]
            else:
                cur_path = [*paths[step-1,ind[0],ind[1]], "NC-{}".format(cur_p)]

            dp_map[step,cur_c,cur_p] = tmp_score[ind]
            paths[step,cur_c,cur_p] = cur_path

    best_path_ind = np.unravel_index(np.argmax(dp_map[-1], axis=None), dp_map[-1].shape)
    best_path = paths[-1, best_path_ind[0], best_path_ind[1]]
    best_pps = get_best_pp_from_path(best_path)
    return best_pps


def get_path_via_DP(nms_pps, nms_scs, nms_tss, use_max=False, plot=False, debug=False):
    final_pps, final_scs, final_tss = [], [], []
    ncc = min(15, len(nms_pps))
    for nc in range(5,ncc+1,3):

        clf = mixture.GaussianMixture(n_components=nc, covariance_type='full')
        clf.fit(nms_pps)
        Z = np.asarray(clf.predict(nms_pps))

        clusters, c_scores, c_tss = {}, {}, {}
        for iz,z in enumerate(Z):
            if not str(z) in clusters.keys():
                clusters[str(z)], c_scores[str(z)], c_tss[str(z)] = [], [], []
                clusters[str(z)].append(iz)
                c_scores[str(z)].append(float(nms_scs[iz]))
                c_tss[str(z)].append(nms_tss[iz])
            if debug: print(clusters.keys())

            selected = []
            for ii in range(nc):
                if str(ii) not in clusters.keys(): continue
                if use_max:
                    ms = max(c_scores[str(ii)])
                    mi = [i for i, j in enumerate(c_scores[str(ii)]) if j == ms]
                    selected.append(clusters[str(ii)][mi[0]])
                else:
                    selected.append(random.choice(clusters[str(ii)]))
                    nms_pps, nms_scs = np.vstack(nms_pps), np.vstack(nms_scs)
        selected = np.asarray(selected)

        cand_pps, cand_scs = nms_pps[selected], nms_scs[selected]
        cand_tss = [nms_tss[si] for si in selected]
        best_idx = apply_dp(cand_pps, cand_scs, 0.1, debug=debug)
        best_pps = [cand_pps[bpi] for bpi in best_idx]
        best_scs = [cand_scs[bpi] for bpi in best_idx]
        best_tss = [cand_tss[bpi] for bpi in best_idx]
        final_pps.append(best_pps)
        final_scs.append(best_scs)
        final_tss.append(best_tss)

    return final_pps, final_scs, final_tss
