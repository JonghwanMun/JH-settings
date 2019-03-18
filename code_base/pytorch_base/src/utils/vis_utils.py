import os
from copy import deepcopy
import itertools

import numpy as np
from PIL import Image
from textwrap import wrap
import matplotlib.pyplot as plt
# no X forwarding on remote machine using ssh & screen & tmux
plt.switch_backend("agg")
from matplotlib import gridspec
from scipy.misc import imresize
from scipy.ndimage.filters import convolve, gaussian_filter
from collections import OrderedDict

import torch

from src.utils import utils, io_utils, net_utils

try:
    import seaborn as sns
    sns.set_style("whitegrid", {"axes.grid": False})
except ImportError:
    print("Install seaborn to colorful visualization!")
except:
    print("Unknown error")

FONTSIZE  = 5

""" helper functions for visualization """
def resize_attention(attention, width=448, height=448,
                     interp="bilinear", use_smoothing=False):
	resized_attention = imresize(attention, [height, width], interp=interp) / 255
	if use_smoothing:
		resized_attention = gaussian_filter(resized_attention, 15)
	return resized_attention

def overlay_attention_on_image(img, attention):
    im = np.asarray(img, dtype=np.float)
    resized_attention = resize_attention(
            attention, width=im.shape[1], height=im.shape[0],
            interp="bilinear", use_smoothing=True)
    im = im * resized_attention[:,:,np.newaxis]
    new_image = Image.fromarray(np.array(im, dtype=np.uint8))
    return new_image

def add_text_to_figure(fig, gc, row, col, row_height, col_width,
                       texts, rotation=-45, colortype=None,
                       y_loc=0.5, fontsize=FONTSIZE):
    try:
        if colortype == "sequence":
            color = sns.color_palette("Set1", n_colors=23, desat=.4)
        else:
            color = ["black" for _ in range(col)]
    except:
        color = ["black" for _ in range(col)]

    for i, text in enumerate(texts):
        sub = fig.add_subplot(gc[row:row+row_height, col+i:col+i+col_width])

        if col == 0:  # location of index(first col of each row)
            sub.text(0.5, y_loc, "\n".join(wrap(text, 8)), ha="center", va="center",
                     rotation=rotation, fontsize=fontsize, wrap=True)
        else:
            sub.text(0.5, y_loc, text, ha="center", va="center", fontsize=fontsize,
                     rotation=rotation, wrap=True, color=color[i%23])
        sub.axis("off")

def add_image_to_figure(fig, gc, row, col, row_height, col_width, images,
                        show_colorbar=False, vmin=None, vmax=None):
    for i, image in enumerate(images):
        sub = fig.add_subplot(gc[row:row+row_height, col:col+col_width])
        cax = sub.imshow(image, vmin=vmin, vmax=vmax)
        sub.axis("off")
        if show_colorbar:
            fig.colorbar(cax)

def add_vector_to_figure(fig, gc, row, col, row_height, col_width, vectors,
                         class_name=None, ymin=0, ymax=0.1, colortype=None):
    try:
        if colortype == "sequence":
            color = sns.color_palette("Set1", n_colors=23, desat=.4)
        else:
            color = None
    except:
        color = None

    for i, vector in enumerate(vectors):
        sub = fig.add_subplot(gc[row:row+row_height, col:col+col_width])
        #sub.set_ylim([ymin, ymax])
        cax = sub.bar(np.arange(vector.shape[0]), vector, width=0.8, color=color)
        sub.axes.get_xaxis().set_visible(False)
        sub.tick_params(axis="y", which="major", labelsize=3)
        if (class_name != None) and ((i+1) == len(vectors)):
            tick_marks = np.arange(len(class_name))
            sub.axes.get_xaxis().set_visible(True)
            plt.setp(sub, xticks=tick_marks, xticklabels=class_name)
            plt.setp(sub.get_xticklabels(), fontsize=2, rotation=45)

def add_question_row_subplot(fig, gc, question, row, col_width=-1):
    if col_width != -1:
        question_width = col_width*2 + 6
    else:
        question_width = 9
    add_text_to_figure(fig, gc, row, 0, 1, 1, ["question"])
    add_text_to_figure(fig, gc, row, 1, 1, question_width,
                       question, rotation=0, colortype="sequence")

def add_attention_row_subplot(fig, gc, img, atts, num_stacks, row, col_width=2):
    for i in range(num_stacks):
        ith_att_weight = atts[i]
        overlayed_att= overlay_attention_on_image(img, ith_att_weight)
        r_pos = row + (i * col_width)

        add_text_to_figure(fig, gc, r_pos, 0, col_width, 1, ["stack_%d" % (i+1)])
        add_image_to_figure(fig, gc, r_pos, 1, col_width, col_width, [img])
        add_image_to_figure(fig, gc, r_pos, col_width+1, col_width,
                            col_width, [overlayed_att])
        add_vector_to_figure(fig, gc, r_pos, col_width*2+1+1, int(col_width/2),
                             col_width, [ith_att_weight.view(14*14).numpy()])
        add_text_to_figure(fig, gc, r_pos+int(col_width/2),
                col_width*2+1+1, int(col_width/2), col_width,
                ["max ({:.6f})".format(ith_att_weight.max()), \
                    "min ({:.6f})".format(ith_att_weight.min())])

def add_answer_row_subplot(fig, gc, answer_logit, gt, itoa, row,
                           class_name=None, is_vqa=False):
    # add ground truth answer
    add_text_to_figure(fig, gc, row, 0, 1, 1, ["GT: {}".format(gt)])

    if type(answer_logit) is list:
        add_text_to_figure(fig, gc, row, 2, 1, 2, \
                ["Assignment: {}".format(" | ".join(str(answer_logit[1][i]+1) \
                    for i in range(answer_logit[1].size(0))))], rotation=0)
        for logit in answer_logit[0]:
            # compute probability of answers
            logit = logit.numpy() # tensor to numpy
            answer_prob = np.exp(logit) / (np.exp(logit).sum() + 1e-10)
            top5_predictions = \
                    ["{}\n({:.3f})".format(itoa[str(a)], answer_prob[a])
                        for i, a in enumerate(answer_prob.argsort()[::-1][:5])]

            add_text_to_figure(fig, gc, row+1, 0, 1, 1, top5_predictions,
                               y_loc=0.5, colortype="sequence")
            if not is_vqa:
                add_vector_to_figure(fig, gc, row+1, 6, 1, 3,
                                     [answer_prob], class_name=class_name)
                add_vector_to_figure(fig, gc, row+1, 10, 1, 3,
                                     [logit], class_name=class_name)
            row += 1
    else:
        # compute probability of answers
        answer_logit = answer_logit.numpy() # tensor to numpy
        answer_prob = np.exp(answer_logit) / (np.exp(answer_logit).sum() + 1e-10)
        top5_predictions = \
                ["{}\n({:.3f})".format(itoa[str(a)], answer_prob[a])
                    for i, a in enumerate(answer_prob.argsort()[::-1][:5])]

        add_text_to_figure(fig, gc, row, 1, 1, 1, top5_predictions,
                           y_loc=0.5, colortype="sequence")

def add_timelines(y, xstart, xstop, color='b'):
    """Plot timelines at y from xstart to xstop with given color."""
    plt.hlines(y, xstart, xstop, color, lw=2)
    plt.vlines(xstart, y+0.007, y-0.007, color, lw=2)
    plt.vlines(xstop, y+0.007, y-0.007, color, lw=2)


def draw_instance_positive_proposals(instance, prop_type="proposal_labels"):

    dr = instance["duration"]
    vid = instance["video_id"]
    vid_labels = instance[prop_type]
    nfeats, K = vid_labels.shape
    # get classnames and featstamps for GT segments
    y_names = ["GT_{}".format(i+1) for i,x in enumerate(instance["gt_times"])]
    featstamps = [utils.timestamp_to_featstamp(x, nfeats, dr) for x in instance["gt_times"]]
    featstamps_for_proposals = utils.get_candidate_featstamps(nfeats, K)
    for n in range(nfeats):
        find = False
        nth_featstamps = deepcopy(featstamps)
        nth_y_names = deepcopy(y_names)
        for k in range(K):
            if vid_labels[n,k] == 1:
                nth_featstamps.append(featstamps_for_proposals[n][k])
                nth_y_names.append("{}_{}".format(n,k))
                find = True
        if find:
            nth_y_names = np.asarray(nth_y_names)

            # get y-values and unique labels
            uniq_names, uniq_idx, idx = np.unique(nth_y_names, True, True)
            y = (idx + 1) / float(len(uniq_names) + 1)

            # draw durations of each segment
            print("Start drawing timestamps ({})".format(n+1))
            colors = sns.color_palette("Set1", n_colors=len(uniq_names), desat=.4)
            for fs, y_, i in zip(nth_featstamps, y, idx):
                add_timelines(y_, fs[0], fs[1], color=colors[i])

            # set x-, y-axis
            ax = plt.gca()
            plt.yticks(y[uniq_idx], uniq_names)
            plt.ylim(0,1)
            plt.xlim(0-nfeats/50.0, nfeats+nfeats/50.0)
            plt.xlabel("Time")
            #plt.savefig("/Users/jonghwan.mun/figs/{:05d}.jpg".format(n))
            #print("saved in ~/figs/{:05d}.jpg".format(n))
            plt.show()
    wait = input("Waiting to show plots")
    plt.clf()
            #ipdb.set_trace()

def visualize_topk_proposals(config, sample_proposals, gts, topk, prefix, mode):
    duration = gts["duration"]
    gt_times = gts["gt_times"]
    prop_timestamps = utils.get_timestamps_for_topk_proposals(
        sample_proposals, duration, topk, apply_nms=True)

    # prepare data
    names = ["GT_{}".format(i+1) for i,x in enumerate(gt_times)]
    timestamps = [gtt for gtt in gt_times]
    for i,ts in enumerate(prop_timestamps):
        tiou, gtidx = utils.iou(ts, gt_times, return_index=True)
        names.append("prop{:02d}_e{:.5f}_t{:.5f}_{:02d}".format(
                i, ts[2], tiou, gtidx+1))
        timestamps.append([ts[0], ts[1]])

    # get y-values and unique labels
    names = np.asarray(names)
    y = np.arange(len(names)+1)[::-1] / float(len(names)+1)

    # Draw proposals
    fig = plt.figure(figsize=(5,5))
    colors = sns.color_palette("Set1", n_colors=len(names), desat=.4)
    for ts, y_, i in zip(timestamps, y, range(len(names))):
        add_timelines(y_, ts[0], ts[1], color=colors[i])

    # set x-, y-axis
    ax = plt.gca()
    plt.yticks(y, names, fontsize=5)
    plt.ylim(0,1)
    plt.xlim(0-duration/50.0, duration+duration/50.0)
    plt.xlabel("Time")

    # save figure
    save_dir = os.path.join(
        config["misc"]["result_dir"], "qualitative", mode)
    save_path = os.path.join(save_dir, prefix + "_proposals.png")
    io_utils.check_and_create_dir(save_dir)
    plt.savefig(save_path, bbox_inches="tight", dpi=450)
    print("Qualtitative result of Topk proposals saved in {}".format(save_path))
    plt.close()
