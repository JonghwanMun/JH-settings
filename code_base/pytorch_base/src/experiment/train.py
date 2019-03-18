import os
import gc
import json
import argparse
import numpy as np
import torch

from src.model import building_networks
from src.experiment import common_functions as cmf
from src.utils import accumulator, timer, utils, io_utils, net_utils

# for reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

""" Get parameters """
def _get_argument_params():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--config_path",
        default="src/experiment/options/default.yml", help="Path to config file.")
	parser.add_argument("--model_type",
        default="ensemble", help="Model type among [||].")
	parser.add_argument("--dataset",
        default="clevr", help="Dataset to train models [|].")
	parser.add_argument("--num_workers", type=int,
        default=4, help="The number of workers for data loader.")
	parser.add_argument("--debug_mode" , action="store_true", default=False,
		help="Train the model in debug mode.")

	params = vars(parser.parse_args())
	print(json.dumps(params, indent=4))
	return params

""" Training the network """
def train(config):

    # create loggers
    it_logger = cmf.create_logger(config, "ITER", "train.log")
    epoch_logger = cmf.create_logger(config, "EPOCH", "scores.log")

    """ Prepare data loader and model"""
    dsets, L = cmf.get_loader(dataset, split=["train", "test"],
                              loader_configs=[config["train_loader"], config["test_loader"]],
                              num_workers=config["misc"]["num_workers"])
    sample_data = dsets["train"].get_samples(1)
    net, start_epoch = cmf.factory_model(config, M, dsets["train"], it_logger)

    # Prepare tensorboard
    net.create_tensorboard_summary("./tensorboard")

    """ Run training network """
    ii = 1
    tm, epoch_tm = timer.Timer(), timer.Timer() # tm: timer
    eval_after = config["evaluation"].get("evaluate_after", 1) # epoch
    eval_every = config["evaluation"].get("every_eval", 1) # epoch
    vis_every = config["misc"].get("vis_every", 1000) # iteration
    print_every = config["misc"].get("print_every", 1) # iteration

    # We evaluate initialized model
    #cmf.test(config, L["test"], net, 0, epoch_logger, mode="Valid")
    for epoch in range(start_epoch, config["optimize"]["num_epoch"]+1):
        net.train_mode() # set network as train mode
        net.reset_status() # initialize status

        for batch in L["train"]:
            # Forward and update the network
            data_load_duration = tm.get_duration()
            tm.reset()
            net_inps, gts = net.prepare_batch(batch)
            outputs = net.forward_update(net_inps, gts)
            run_duration = tm.get_duration()

            # Compute status for current batch: loss, evaluation scores, etc
            net.compute_status(outputs["net_output"][0], gts)

            # print learning status
            if ii % print_every == 0:
                net.print_status(epoch)
                lr = net_utils.adjust_lr(net.it,
                        net.it_per_epoch, net.config["optimize"])
                txt = "fetching for {:.3f}s, optimizing for {:.3f}s, lr = {:.5f}"
                it_logger.info(txt.format(data_load_duration, run_duration, lr))

            # check results for pre-selected samples over training
            if vis_every > 0 and (ii % vis_every == 0):
                net.save_results(sample_data,
                        "iteration_{}".format(ii), mode="Train")

            ii += 1
            tm.reset()

            if config["misc"]["debug"] and (ii > 2):
                break
            # iteration done

        # print training time for 1 epoch
        txt = "[Epoch {}] total time of training 1 epoch: {:.3f}s"
        it_logger.info(txt.format(epoch, epoch_tm.get_duration()))

        # save network every epoch
        ckpt_path = os.path.join(config["misc"]["result_dir"],
                                 "checkpoints", "epoch_{:03d}.pkl".format(epoch))
        net.save_checkpoint(ckpt_path)

        # save results (predictions, visualizations)
        # Note: save_results() should be called before print_counters_info()
        net.save_results(sample_data, "epoch_{:03d}".format(epoch), mode="Train")

        # print status (metric) accumulated over each epoch
        net.print_counters_info(epoch, epoch_logger, mode="Train")

        # validate network
        if (epoch >= eval_after) and (epoch % eval_every == 0):
            cmf.test(config, L["test"], net, epoch, epoch_logger, mode="Valid")

        # check curriculum learning
        net.check_apply_curriculum(epoch)

        # reset reference time to compute duration of loading data
        tm.reset(); epoch_tm.reset()
        # epoch done

if __name__ == "__main__":
    # get parameters from cmd
    params = _get_argument_params()
    global M, dataset
    M, dataset, config = cmf.prepare_experiment(params)

    # train network
    train(config)
