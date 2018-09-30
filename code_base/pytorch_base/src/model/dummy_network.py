import os
import ipdb
import copy
import json
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from src.model import building_blocks
from src.model.virtual_network import AbstractNetwork
from src.utils import accumulator, utils, io_utils, vis_utils, net_utils

class Dummy(AbstractNetwork):
    def __init__(self, config, verbose=True):
        super(Dummy, self).__init__(config)

        self.net = building_blocks.Dummy(config)
        self.criterion = OrderedDict()
        self.criterion["weighted_bce"] = building_blocks.WeightedBCE()

        self.build_network(config)
        self.build_evaluator(config)

        # set models to update
        self.model_list = ["net", "criterion"]
        self.models_to_update = ["net", "criterion"]

    def forward(self, inp):
        crit_inp, misc = OrderedDict(), OrderedDict()
        crit_inp["net_output"], misc["net_misc"] = self.dummy(inp)
        return [crit_inp, misc]

    def reset_status(self, init_reset=False):
        """ Reset (initialize) metric scores or losses (status).
        """
        if self.status == None:
            self.status = OrderedDict()
            self.status["total_loss"] = 0
        else:
            for k in self.status.keys():
                self.status[k] = 0

    def attach_predictions(self):
        pass

    def save_predictions(self, prefix, mode):
        pass

    """ methods for counters """
    def _create_counters(self):
        self.counters = OrderedDict()
        self.counters["total_loss"] = accumulator.Accumulator("loss")

    def bring_loader_info(self, dataset):
        pass

    """ methods for metrics """
    def compute_metrics(self, logits, gts):
        pass

    """ method for status (metrics) """
    def compute_status(self, logits, gts):
        pass

    @classmethod
    def model_specific_config_update(cls, config):
        print("You would need to implement 'model_specific_config_update'")
        return config

    @staticmethod
    def override_config_from_loader(config, loader):
        # model: question embedding layer
        config["model"]["vocab_size"] = loader.get_vocab_size()
        config["model"]["word_emb_padding_idx"] = loader.get_idx_empty_word()
        # model: classifier and assignment layer
        config["model"]["num_labels"] = loader.get_num_answers()

        return config


