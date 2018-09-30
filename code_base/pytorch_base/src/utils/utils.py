import pdb
import random
import string
import numpy as np

import torch
from src.evaluation.eval_densecap import ANETcaptions
from src.utils import net_utils

FONTSIZE  = 5

""" Methods for string """
def tokenize(txt, translator=None):
    """ Tokenize text - converting to lower characters, eliminating puncutations
    Args:
        txt: text to be tokenized; str
        translator: this includes punctuations for translate() func; dict()
    """
    if not translator:
        translator = str.maketrans("", "", string.punctuation)
    tokens = str(txt).lower().translate(translator).strip().split()
    return tokens

def label2string(itow, label, start_idx=2, end_idx=3):
    """ Convert labels to string (question, caption, etc)
    Args:
        itow: dictionry for mapping index to word; dict()
        label: index of labels
    """
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy().squeeze()
    if label.ndim == 0:
        if label == end_idx:
            return "abcdefghi"
        else:
            return itow[str(label)]
    assert label.ndim == 1, "{}".format(label.ndim)
    txt = []
    for l in label:
        if l == start_idx: continue
        if l == end_idx: break
        else: txt.append(itow[str(l)])
    return " ".join(txt).strip()

def string2label(itow, txt):
    """ Convert string (question, caption, etc) to labels
    Args:
        wtoi: dictionry for mapping word to index; dict()
        txt: index of labels; list()
    """
    if len(txt) == 0:
        return None
    else:
        label = []
        for w in txt:
            label.append(itow[w])
        return np.asarray(label)


""" Methods for dictionary """
def shuffle_dict(dic):
    keys = list(dic.keys())
    idx = np.arange(len(dic[keys[0]]))
    random.shuffle(idx)
    new_dic = {k:[] for k in keys}
    for ii in idx:
        for k in keys:
            new_dic[k].append(dic[k][ii])
    return new_dic

def get_value_from_dict(dic, key, default_value=None):
    """ Get value from dicionary (if key don"t exists, use default value)
    Args:
        dic: dictionary
        key: key value
        default_value: default_value
    Returns:
        dic[key] or default_value
    """
    if dic == None: return default_value

    if key in dic.keys(): return dic[key]
    else: return default_value


""" Methods for path """
def get_filename_from_path(file_path, delimiter="/"):
    """ Get filename from file path (filename.txt -> filename)
    """
    filename = file_path.split(delimiter)[-1]
    return filename.split(".")[0]


""" Methods for proposals """
def tiou(interval_1, interval_2):
    start_i, end_i = interval_1[0], interval_1[1]
    start, end = interval_2[0], interval_2[1]
    intersection = max(0, min(end, end_i) - max(start, start_i))
    union = min(max(end, end_i) - min(start, start_i), end-start + end_i-start_i)
    tiou = float(intersection) / (union + 1e-8)
    return tiou

def iou(interval, featstamps, return_index=False):
    """
    Measures temporal IoU
    """
    start_i, end_i = interval[0], interval[1]
    best_iou = 0.0
    gt_index = -1
    for i, (start, end) in enumerate(featstamps):
        intersection = max(0, min(end, end_i) - max(start, start_i))
        union = min(max(end, end_i) - min(start, start_i), end - start + end_i - start_i)
        overlap = float(intersection) / (union + 1e-8)
        if overlap >= best_iou:
            best_iou = overlap
            gt_index = i
    if return_index:
        return best_iou, gt_index
    return best_iou

def timestamp_to_featstamp(timestamp, nfeats, duration):
    """
    Function to measure 1D overlap
    Convert the timestamps to feature indices
    """
    start, end = timestamp
    start = min(int(round(start / duration * nfeats)), nfeats - 1)
    end = max(int(round(end / duration * nfeats)), start + 1)
    return start, end

def get_candidate_featstamps_given_K_at(end_pos, K):
    distance = np.arange(0, K)
    featstamps = [[end_pos-d, end_pos+1] for d in distance]
    return featstamps

def get_candidate_featstamps(num_steps, K, flat=False):
    featstamps = []
    for i in range(num_steps):
        if flat:
            featstamps.extend(get_candidate_featstamps_given_K_at(i, K))
        else:
            featstamps.append(get_candidate_featstamps_given_K_at(i, K))
    return featstamps

def get_candidate_timestamps(duration, num_steps, num_props, flat=True):
    step_length = duration / num_steps
    timestamps = []
    for time_step in np.arange(num_steps):
        if not flat: timestamps.append([])

        end = (time_step+1) * step_length
        for k in np.arange(num_props):
            start = max(0, time_step - k) * step_length
            if flat:
                timestamps.append((start, end))
            else:
                timestamps[time_step].append((start, end))
    return timestamps

def get_timestamps_for_topk_proposals(proposals, duration, topk=None, apply_nms=False):
    # Here, we assume that proposals for only one video are given as an input
    # if topk is None, extract all possible proposals
    _, L, K = proposals.size() # [1, L, K]

    timestamps = get_candidate_timestamps(duration, L, K, flat=True)
    if apply_nms:
        _, val, idx = nms_detections(timestamps, proposals.view(L*K),
                overlap=0.80, topk=1000, return_index=True)
        if len(val) > topk:
            val, idx = val[:topk], idx[:topk]
    else:
        if topk and topk < L * K:
            # keep only topk proposals
            val, idx = proposals.view(L*K).sort(descending=True)
            val, idx = val[:topk], idx[:topk]
        else:
            val, idx = proposals.view(L*K), np.arange(L*K)

    topk_timestamps = []
    for score,i in zip(val,idx):
        topk_timestamps.append([timestamps[i][0], timestamps[i][1], score])

    return topk_timestamps

def nms_detections(props, scores, overlap=0.7, captions=None, topk=-1, return_index=False):
    """ Non-maximum suppression: Greedily select high-scoring detections and
    skip detections that are significantly covered by a previously selected
    detection. This version is translated from Matlab code by Tomasz
    Malisiewicz, who sped up Pedro Felzenszwalb's code.
    Args:
        props: ndarray
            Two-dimensional array of shape (num_props, 2), containing the start and
            end boundaries of the temporal proposals.
        scores: ndarray
            One-dimensional array of shape (num_props,), containing the corresponding
            scores for each detection above.
    Returns:
        nms_props, nms_scores, (nms_caps) : ndarrays, ndarrays, strs
            Arrays with the same number of dimensions as the original input, but
            with only the proposals selected after non-maximum suppression.
    """
    if isinstance(props, torch.Tensor): props = net_utils.to_data(props)
    if isinstance(props, list): props = np.asarray(props)
    if isinstance(scores, torch.Tensor): scores = net_utils.to_data(scores)
    t1 = props[:, 0]
    t2 = props[:, 1]
    ind = np.argsort(scores)
    if topk > 0 and len(ind) > topk:
        ind = ind[-topk:]
    area = (t2 - t1 + 1).astype(float)
    pick = []
    while len(ind) > 0:
        i = ind[-1]     # index with highest proposal score
        pick.append(i)
        ind = ind[:-1]
        tt1 = np.maximum(t1[i], t1[ind])
        tt2 = np.minimum(t2[i], t2[ind])
        wh = np.maximum(0., tt2 - tt1 + 1.0)
        o = wh / (area[i] + area[ind] - wh)
        ind = ind[np.nonzero(o <= overlap)[0]]
    if captions:
        nms_props, nms_scores, nms_caps = props[pick, :], scores[pick], captions[pick]
        if return_index:
            return nms_props, nms_scores, nms_caps, pick
        else:
            return nms_props, nms_scores, nms_caps
    else:
        nms_props, nms_scores = props[pick, :], scores[pick]
        if return_index:
            return nms_props, nms_scores, pick
        else:
            return nms_props, nms_scores

def nms_all_predictions(preds, overlap=0.7):
    new_preds = {}
    for vid,pred in tqdm(preds.items()):
        pps, scs, new_preds[vid] = [], [], []
        for p in pred:
            pps.append(np.array(p["timestamp"]))
            scs.append(np.array(p["eventness"]))
        new_pp, new_sc = nms_detections(np.asarray(pps), np.asarray(scs), overlap)
        for npp,nsc in zip(new_pp,new_sc):
            new_preds[vid].append({"timestamp": npp, "eventness": nsc})
    return new_preds


""" Methods related to evalutation """
def get_proposal_evaluator(gt_path, tious, max_proposals):
    return ANETcaptions(ground_truth_filenames=gt_path, prediction_filename=None,
        tious=tious, max_proposals=max_proposals, verbose=False)

def get_densecap_evaluator(gt_path, tious, max_proposals):
    return ANETcaptions(ground_truth_filenames=gt_path, prediction_filename=None,
        tious=tious, max_proposals=max_proposals, verbose=True)
