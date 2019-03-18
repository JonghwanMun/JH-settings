import os
import json
import argparse

from src.model import building_networks
from src.experiment import common_functions as cmf
from src.utils import accumulator, timer, utils, io_utils
from src.utils.tensorboard_utils import PytorchSummary

""" Get parameters """
def _get_argument_params():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--loader_config_path", default="src/experiment/options/test_loader_config.yml",
            help="Do evaluation or getting/saving some values")
	parser.add_argument("-exp", "--experiment", type=str, required=True, help="Experiment or configuration name")
	parser.add_argument("-m", "--model_type", default="densecap", help="Model type among [san | ensemble | saaa].")
	parser.add_argument("-d", "--dataset", default="densecap", help="dataset to train models [clevr|vqa].")
	parser.add_argument("-nw", "--num_workers", type=int, default=4, help="The number of workers for data loader.")
	parser.add_argument("-se", "--start_epoch", type=int, default=10, help="Start epoch to evaluate.")
	parser.add_argument("-ee", "--end_epoch", type=int, default=50, help="End epoch to evaluate.")
	parser.add_argument("-es", "--epoch_stride", type=int, default=5, help="Stride for jumping epoch.")
	parser.add_argument("-server", "--test_on_server" , action="store_true", default=False,
            help="Evaluate on test set of evaluation server")
	parser.add_argument("-gt", "--evaluate_on_gt" , action="store_true", default=False,
            help="Evaluate on Ground-Truth")
	parser.add_argument("-top", "--evaluate_on_top1000" , action="store_true", default=False,
            help="Evaluate on top 1000 proposals")
	parser.add_argument("-pp", "--proposal", type=str, default="", help="Experiment or configuration name")
	parser.add_argument("--debug_mode" , action="store_true", default=False,
            help="Run the script in debug mode")

	params = vars(parser.parse_args())
	print (json.dumps(params, indent=4))
	return params

def main(params):
    # Obtain configuration path
    exp_path = os.path.join("results", params["dataset"],
                            params["model_type"], params["experiment"])
    config_path = os.path.join(exp_path, "config.yml")
    params["config_path"] = config_path

    # prepare model and dataset
    M, dataset, config = cmf.prepare_experiment(params)

    # evaluate on GT
    config["evaluation"]["use_gt"] = params["evaluate_on_gt"]

    # evaluate on Top 1000 proposals
    if params["evaluate_on_top1000"]:
        config["evaluation"]["use_gt"] = False
        config["evaluation"]["apply_nms"] = False

    if len(params["proposal"]) > 0:
        config["evaluation"]["precomputed_proposal_sequence"] = params["proposal"]

    # create logger
    epoch_logger = cmf.create_logger(config, "EPOCH", "test.log")

    """ Build data loader """
    loader_config = io_utils.load_yaml(params["loader_config_path"])
    if params["test_on_server"]:
        loader_config = loader_config["test_loader"]
        test_on = "Test_Server"
    else:
        loader_config = loader_config["val_loader"]
        test_on = "Test"
    dsets, L = cmf.get_loader(dataset, split=["test"],
                              loader_configs=[loader_config],
                              num_workers=params["num_workers"])
    config = M.override_config_from_dataset(config, dsets["test"], mode="Test")
    config["model"]["resume"] = True
    tensorboard_path = config["misc"]["tensorboard_dir"]
    config["misc"]["tensorboard_dir"] = "" #
    config["misc"]["debug"] = params["debug_mode"]

    """ Evaluating networks """
    e0 = params["start_epoch"]
    e1 = params["end_epoch"]
    es = params["epoch_stride"]
    io_utils.check_and_create_dir(tensorboard_path + "_test_s{}_e{}".format(e0,e1))
    summary = PytorchSummary(tensorboard_path + "_test_s{}_e{}".format(e0,e1))
    for epoch in range(e0, e1+1, es):
        """ Build network """
        config["model"]["checkpoint_path"] = \
            os.path.join(exp_path, "checkpoints", "epoch_{:03d}.pkl".format(epoch))
        net, _ = cmf.factory_model(config, M, dsets["test"], None)
        net.set_tensorboard_summary(summary)

        cmf.test(config, L["test"], net, epoch, None, epoch_logger, on=test_on)

if __name__ == "__main__":
    params = _get_argument_params()
    main(params)
