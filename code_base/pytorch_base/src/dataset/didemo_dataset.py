"""
Base data Loader.
Revise this file to use your own data.

Written by Jonghwan Mun
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import json
import h5py
import string
import numpy as np
np.set_printoptions(precision=4)
from tqdm import tqdm
from copy import deepcopy
from itertools import product
from abc import abstractmethod
from collections import defaultdict, OrderedDict

import torch
import torch.utils.data as data
from torch.utils.data import Dataset

from src.dataset.abstract_dataset import AbstractDataset
from src.utils import utils, io_utils

def create_loaders(split, loader_configs, num_workers):
    dsets, L = {}, {}
    for di,dt in enumerate(split):
        dsets[dt] = BaseDataset(loader_configs[di])
        L[dt] = data.DataLoader(
            dsets[dt],
            batch_size = loader_configs[di]["batch_size"],
            num_workers = num_workers,
            shuffle = True,
            collate_fn = dsets[dt].collate_fn
        )
    return dsets, L


class BaseDataset(AbstractDataset):
    def __init__(self, config):
        super(self.__class__, self).__init__(config)


        # Get paths for annotations and features, etc
        self._get_data_path(config)

        # Add own codes
        if not self._exist_data():
            self.generate_labels(config)

        # Some variables
        self.batch_size = config.get("batch_size", 1)
        self.num_instances = len(self.vid_ids)

    def __getitem__(self, idx):
        instance = {
            "dummy": True
        }

        return instance

    def _get_data_path(self, config):
        self.paths = {
            "dummy_path": True,
        }

    def _exist_data(self):
        pass

    def collate_fn(self, data):
        batch = {key: [d[key] for d in data] for key in data[0]}

        if len(data) == 1:
            for k,v in batch.items():
                if k in ["features", "proposal_labels"]:
                    batch[k] = torch.cat(batch[k], 0)
                else:
                    batch[k] = batch[k][0]
        else:
            for key in ["features", "proposal_labels"]:
                batch[key] = torch.cat(batch[key], 0)
        return batch

# for debugging
def get_loader():
    conf = {
        "train_loader": {
            "option": True,
        }
    }
    print(json.dumps(conf, indent=4))
    dsets, L = create_loaders(["train"], [conf["train_loader"]], num_workers=4)
    return dsets["train"]

if __name__ == "__main__":
    dset = get_loader()
    draw_instance(dset)
    ipdb.set_trace()
