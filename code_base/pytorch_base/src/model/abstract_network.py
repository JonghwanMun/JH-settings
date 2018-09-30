import os
import pdb
import json
import numpy as np
from abc import abstractmethod
from collections import OrderedDict

import torch
import torch.nn as nn

from src.utils import accumulator, timer, utils, io_utils, net_utils
from src.utils.tensorboard_utils import PytorchSummary

class AbstractNetwork(nn.Module):
    def __init__(self, config, logger=None, verbose=False):
        super(AbstractNetwork, self).__init__() # Must call super __init__()

        self.optimizer, self.sample_data, self.models_to_update = None, None, None
        self.training_mode = True
        self.evaluate_after = config["evaluation"].get("evaluate_after", 1)

        self.it = 0 # it: iteration
        self.tm = timer.Timer() # tm: timer
        self.grad_clip = config["optimize"].get("gradient_clip", 10)
        self.update_every = config["optimize"].get("update_every", 1)
        self.use_gpu = config["model"].get("use_gpu",
                True if torch.cuda.is_available else False)
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        if len(config["misc"]["tensorboard_dir"]) > 0:
            self.create_tensorboard_summary(config["misc"]["tensorboard_dir"])

        # save configuration for later network reproduction
        resume = config["model"].get("resume", False)
        if not resume:
            save_config_path = os.path.join(
                config["misc"]["result_dir"], "config.yml")
            io_utils.write_yaml(save_config_path, config)
        self.config = config
        if logger is not None:
            self.log = logger.info
            logger.info(json.dumps(config, indent=2))
        else:
            self.log = print

    """ methods for forward/backward """
    @abstractmethod
    def forward(self, net_inps):
        """ Forward network
        Args:
            net_inps: inputs for network; dict()
        Returns:
            [crit_inp, misc]: two items of list
                - inputs for criterion; dict()
                - intermediate values for visualization, etc; dict()
        """
        pass

    def loss_fn(self, crit_inp, gts, count_loss=True):
        """ Compute loss
        Args:
            crit_inp: inputs for criterion which is outputs from forward(); dict()
            gts: ground truth
            count_loss: flag of accumulating loss or not (training or inference)
        Returns:
            loss: results of self.criterion; dict()
        """
        self.loss = self.criterion(crit_inp, gts)
        for name in self.loss.keys():
            self.status[name] = self.loss[name].detach().cpu().numpy()
        if count_loss:
            for name in self.loss.keys():
                self.counters[name].add(self.status[name], 1)
        return self.loss

    def update(self, loss):
        """ Update the network
        Args:
            loss: loss to train the network; dict()
        """
        self.it +=1
        lr = net_utils.adjust_lr(self.it, self.it_per_epoch, self.config["optimize"])

        # initialize optimizer
        if self.optimizer == None:
            self.create_optimizer(lr)
            self.optimizer.zero_grad() # set gradients as zero before update

        total_loss = loss["total_loss"] / self.update_every
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.grad_clip)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        if self.it % self.update_every == 0:
            self.optimizer.step()
            self.optimizer.zero_grad() # set gradients as zero before updating the network

        if self.null_feats is not None:
            for k,v in self.null_feats.items():
                del v

    def forward_update(self, net_inps, gts):
        """ Forward and update the network at the same time
        Args:
            net_inps: inputs for network; dict()
            gts: ground truth; dict()
        Returns:
            {loss, net_output}: two items of dictionary
                - loss: results from self.criterion(); dict()
                - net_output: first output from self.forward(); dict()
        """
        outputs = self.forward(net_inps)
        loss = self.loss_fn(outputs[0], gts, count_loss=True)
        self.update(loss)
        return {"loss":loss, "net_output":outputs}

    def compute_loss(self, net_inps, gts):
        """ Compute loss and network's output at once
        Args:
            net_inps: inputs for network; dict()
            gts: ground truth; dict()
        Returns:
            {loss, net_output}: two items of dictionary
                - loss: results from self.criterion(); dict()
                - net_output: first output from self.forward(); dict()
        """
        outputs = self.forward(net_inps)
        loss = self.loss_fn(outputs[0], gts, count_loss=True)
        return {"loss":loss, "net_output":outputs}

    def create_optimizer(self, lr):
        """ Create optimizer for training phase
        Currently supported optimizer list: [SGD, Adam]
        Args:
            lr: learning rate; int
        """
        opt_type = self.config["optimize"]["optimizer_type"]
        if opt_type == "SGD":
            self.optimizer = torch.optim.SGD(
                    self.get_parameters(), lr=lr,
                    momentum=self.config["optimize"]["momentum"],
                    weight_decay=self.config["optimize"]["weight_decay"])
        elif opt_type == "Adam":
            self.optimizer = torch.optim.Adam(self.get_parameters(), lr=lr)
        elif opt_type == "Adadelta":
            self.optimizer = torch.optim.Adadelta(self.get_parameters(), lr=lr)
        elif opt_type == "RMSprop":
            self.optimizer = torch.optim.RMSprop(self.get_parameters(), lr=lr)
        else:
            raise NotImplementedError(
                "Not supported optimizer [{}]".format(opt_type))

    @abstractmethod
    def build_network(self, config):
        pass

    @abstractmethod
    def build_evaluator(self, config):
        pass

    @abstractmethod
    def prepare_batch(self, batch):
        """ Prepare batch to be used for network
        e.g., shipping batch to gpu
        Args:
            batch: batch data; dict()
        Returns:
            batch: batch; dict()
        """
        pass

    @abstractmethod
    def check_apply_curriculum(self, epoch=-1):
        """ Check and apply curriculum learning
        """
        pass

    """ methods for checkpoint """
    def load_checkpoint(self, ckpt_path, logger=None):
        """ Load checkpoint of the network.
        Args:
            ckpt_path: checkpoint file path; str
        """
        self.log("Checkpoint is loaded from {}".format(ckpt_path))
        model_state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        self.log("[{}] are in checkpoint".format("|".join(model_state_dict.keys())))
        for m in model_state_dict.keys():
            if m == "criterion": continue
            if m in self.model_list:
                self[m].load_state_dict(model_state_dict[m])
                self.log("{} is initialized from checkpoint".format(m))
            else:
                self.log("{} is not in {}".format(m, "|".join(self.model_list)))


    def save_checkpoint(self, ckpt_path, logger=None):
        """ Save checkpoint of the network.
        Args:
            ckpt_path: checkpoint file path
        """
        model_state_dict = {m:self[m].state_dict() for m in self.model_list if m != "criterion"}
        torch.save(model_state_dict, ckpt_path)

        self.log("Checkpoint [{}] is saved in {}".format(
                    " | ".join(model_state_dict.keys()), ckpt_path))

    """ method for status (metrics) """
    def reset_status(self, init_reset=False):
        """ Reset (initialize) metric scores or losses (status).
        """
        if init_reset:
            self.status = OrderedDict()
            self.status["total_loss"] = 0
            for k,v in self.criterion.get_items():
                self.status[k] = 0
        else:
            for k in self.status.keys():
                self.status[k] = 0

    @abstractmethod
    def compute_status(self, logits, gts):
        """ Compute metric scores or losses (status).
            You may need to implement this method.
        Args:
            logits: output logits of network.
            gts: ground-truth
        """
        pass

    def print_status(self, epoch, logger=None, mode="Train", enter_every=2):
        """ Print current metric scores or losses (status).
            You are encouraged to implement this method.
        Args:
            epoch: current epoch
        """
        # print status information
        txt = "epoch {} step {} ".format(epoch, self.it)
        for i,(k,v) in enumerate(self.status.items()):
            if (i+1) % enter_every == 0:
                txt += "{} = {:.4f}, ".format(k, float(v))
                self.log(txt)
                txt = ""
            else:
                txt += "{} = {:.4f}, ".format(k, float(v))
        if len(txt) > 0: self.log(txt)

    """ methods for counters """
    def _create_counters(self):
        self.counters = OrderedDict()
        self.counters["total_loss"] = accumulator.Accumulator("total_loss")
        for k,v in self.criterion.get_items():
            self.counters[k] = accumulator.Accumulator(k)

    def reset_counters(self):
        for k,v in self.counters.items():
            v.reset()

    def print_counters_info(self, epoch, logger, on="Train"):
        if on != "Train" and epoch < self.evaluate_after:
            self.reset_counters()
            return

        txt = "[{}] {} epoch".format(on, epoch)
        for k,v in self.counters.items():
            txt += ", {} = {:.4f}".format(v.get_name(), v.get_average())
        logger.info(txt)

        if self.use_tf_summary:
            self.write_counter_summary(epoch, on)

        # reset counters
        self.reset_counters()


    """ methods for tensorboard """
    def create_tensorboard_summary(self, tensorboard_dir):
        self.use_tf_summary = True
        self.summary = PytorchSummary(tensorboard_dir)
        #self.write_params_summary(epoch=0)

    def set_tensorboard_summary(self, summary):
        self.use_tf_summary = True
        self.summary = summary

    def write_params_summary(self, epoch):
        if self.models_to_update is None:
            for name, param in self.named_parameters():
                self.summary.add_histogram("model/{}".format(name),
                    param, global_step=epoch)
        else:
            for m in self.models_to_update:
                for name, param in self[m].named_parameters():
                    self.summary.add_histogram("model/{}/{}".format(m, name),
                        param, global_step=epoch)

    def write_status_summary(self):
        for k,v in self.status.items():
            self.summary.add_scalar('status/' + k, v, global_step=self.it)

    def write_counter_summary(self, epoch, mode):
        for k,v in self.counters.items():
            self.summary.add_scalar(mode + '/counters/' + v.get_name(),
                               v.get_average(), global_step=epoch)

    @abstractmethod
    def bring_loader_info(self, dataset):
        pass

    """ wrapper methods of nn.Modules """
    def _get_parameter(self, net):
        if isinstance(net, dict):
            for k,v in net.items():
                self._get_parameter(v)
        else:
            for name,param in net.named_parameters():
                yield param

    def get_parameters(self):
        if self.models_to_update is None:
            for name, param in self.named_parameters():
                yield param
        else:
            for m in self.models_to_update:
                if isinstance(self[m], dict):
                    for k,v in self[m].items():
                        for name, param in v.named_parameters():
                            yield param
                else:
                    for name, param in self[m].named_parameters():
                        yield param

    def cpu_mode(self, logger=None):
        sel.log("Setting cpu() for [{}]".format(" | ".join(self.model_list)))
        self.cpu()

    def gpu_mode(self, logger=None):
        #cudnn.benchmark = False
        if torch.cuda.is_available():
            self.log("Setting gpu() for [{}]".format(" | ".join(self.model_list)))
            self.cuda()
        else:
            raise NotImplementedError("Available GPU not exists")

    def train_mode(self, logger=None):
        self.train()
        self.training_mode = True
        self.log("Setting train() for [{}]".format(" | ".join(self.model_list)))

    def eval_mode(self, logger=None):
        self.eval()
        self.training_mode = False
        self.log("Setting eval() for [{}]".format(" | ".join(self.model_list)))

    def _set_sample_data(self, data):
        if self.sample_data == None:
            self.sample_data = copy.deepcopy(data)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    @staticmethod
    def model_specific_config_update(config):
        return config

    @staticmethod
    def override_config_from_dataset(config, loader, mode="Train"):
        if mode == "Test":
            config["model"]["resume"] = True
        return config

