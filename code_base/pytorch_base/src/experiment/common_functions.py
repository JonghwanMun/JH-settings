import os
import time
import h5py
import json
import yaml
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict

import torch
import torch.utils.data as data
import torch.nn.functional as F

from src.dataset import proposal_dataset, densecap_dataset, densecap_proposal_dataset
from src.model import building_blocks, building_networks
from src.utils import accumulator, timer, utils, io_utils, net_utils

""" get base model """
def get_model(model_type):
    if model_type in ["sst", "SST"]:
        M = getattr(building_networks, "SST")
    elif model_type in ["bisst", "BiSST"]:
        M = getattr(building_networks, "BiSST")
    elif model_type in ["densecap", "DENSECAP"]:
        M = getattr(building_networks, "DenseCap")
    elif model_type in ["e2e", "E2E"]:
        M = getattr(building_networks, "E2E_DenseCap")
    elif model_type in ["seq_gen", "SEQGEN"]:
        M = getattr(building_networks, "SequenceGenerator")
    elif model_type in ["mft", "MFT"]:
        M = getattr(building_networks, "MFT")
    else:
        raise NotImplementedError("Not supported model type ({})".format(model_type))
    return M

def get_dataset(dataset):
    if dataset == "proposal":
        D = eval("proposal_dataset")
    elif dataset == "densecap":
        D = eval("densecap_dataset")
    elif dataset == "proposal_densecap":
        D = eval("densecap_proposal_dataset")
    else:
        raise NotImplementedError("Not supported dataset type ({})".format(dataset))
    return D

def get_loader(D, split=[], loader_configs=[], num_workers=2):
    assert len(split) > 0
    assert len(split) == len(loader_configs)
    return D.create_loaders(split, loader_configs, num_workers)

def update_config_from_params(config, params):
    config["misc"]["debug"] = params["debug_mode"]
    config["misc"]["num_workers"] = params["num_workers"]
    config["misc"]["dataset"] = params["dataset"]
    exp_prefix = utils.get_filename_from_path(
            params["config_path"], delimiter="options/") if "options" in params["config_path"] \
            else utils.get_filename_from_path(params["config_path"], delimiter="results/")[:-7]
    config["misc"]["exp_prefix"] = exp_prefix
    config["misc"]["result_dir"] = os.path.join("results", exp_prefix)
    config["misc"]["tensorboard_dir"] = os.path.join("tensorboard", exp_prefix)
    config["misc"]["model_type"] = params["model_type"]
    if not "use_gpu" in config["model"].keys():
        if torch.cuda.is_available():
            config["model"]["use_gpu"] = True
        else:
            config["model"]["use_gpu"] = False

    return config

def prepare_experiment(params):
    M = get_model(params["model_type"])
    D = get_dataset(params["dataset"])

    # loading configuration and setting environment
    config = io_utils.load_yaml(params["config_path"])
    config = update_config_from_params(config, params)
    create_save_dirs(config["misc"])

    return M, D, config

def factory_model(config, M, dset, logger=None):
    config = M.model_specific_config_update(config)
    net = M(config, logger=logger)
    net.bring_loader_info(dset)

    # load checkpoint
    start_epoch = 1
    if config["model"]["resume"]:
        assert len(config["model"]["checkpoint_path"]) > 0
        net.load_checkpoint(config["model"]["checkpoint_path"], logger)
        start_epoch = int(utils.get_filename_from_path(
                config["model"]["checkpoint_path"]).split("_")[-1]) + 1
    net.check_apply_curriculum(start_epoch)

    # ship network to use gpu
    if config["model"]["use_gpu"]: net.gpu_mode(logger)
    if logger is not None: logger.info(net)
    return net, start_epoch

def create_save_dirs(config):
    """ Create neccessary directories for training and evaluating models
    """
	# create directory for checkpoints
    io_utils.check_and_create_dir(os.path.join(config["result_dir"], "checkpoints"))
	# create directory for results
    io_utils.check_and_create_dir(os.path.join(config["result_dir"], "status"))
    io_utils.check_and_create_dir(os.path.join(config["result_dir"], "qualitative"))


""" Get logger """
def create_logger(config, logger_name, log_path):
    logger_path = os.path.join(
            config["misc"]["result_dir"], log_path)
    logger = io_utils.get_logger(
        logger_name, log_file_path=logger_path,\
        print_lev=getattr(logging, config["logging"]["print_level"]),\
        write_lev=getattr(logging, config["logging"]["write_level"]))
    return logger

""" evaluate the network """
def test(config, loader, net, epoch,
         it_logger=None, epoch_logger=None, on="Test"):

    with torch.no_grad():
        net.eval_mode() # set network as evaluation mode
        net.reset_status() # reset status

        """ Testing network """
        ii = 1
        for batch in tqdm(loader, desc="{}".format(on)):
            # forward the network
            net_inps, gts = net.prepare_batch(batch)
            outputs = net(net_inps) # only forward
            """
            if on == "Valid":
                outputs = net.compute_loss(net_inps, gts)
            elif on == "Test":
                outputs = net(net_inps) # only forward
            else:
                raise NotImplementedError()
            """

            # Compute status for current batch: loss, evaluation scores, etc
            net.compute_status(outputs[0], gts, mode=on, logger=it_logger)

            ii += 1
            if config["misc"]["debug"] and (ii > 100):
                break
            # end for batch in loader

        net.save_results(None, "epoch_{:03d}".format(epoch), mode=on)
        net.print_counters_info(epoch, epoch_logger, on=on)

""" Methods for debugging """
def one_step_forward(L, net, logger):
    # fetch the batch
    batch = next(iter(L))

    # forward and update the network
    outputs = net.forward_update(batch)

    # accumulate the number of correct answers
    net.compute_status(outputs, batch["gt"])

    # print learning status
    net.print_status(1, logger)
