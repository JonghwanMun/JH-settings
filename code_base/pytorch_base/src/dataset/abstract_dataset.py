"""
Written by Jonghwan Mun
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from abc import abstractmethod

from torch.utils.data import Dataset

class AbstractDataset(Dataset):
    """
    All dataset loader classes will inherit from this class.
    """

    def __init__(self, config):
        """ Initialize data loader class, to be overwritten. """
        pass

    def __len__(self):
        return self.num_instances

    def __getitem__(self, idx):
        """ To be overwritten. """
        pass

    def get_samples(self, num_samples):
        """ Get smaple instances for visualization during training.
        Args:
            num_samples: number of samples; int
        """
        samples = []
        for i in range(num_samples):
            # randomly select sample index
            idx = np.random.randint(0, len(self)-1)
            sample = self.__getitem__(idx)
            samples.append(sample)

        return self.collate_fn(samples)

    def get_instance(self):
        """ Get a single instances for debugging.  """
        # randomly select sample index
        idx = np.random.randint(0, len(self)-1)
        return self.__getitem__(idx)

    def get_iteration_per_epoch(self):
        """ Get the number of iterations for each epoch given batch size """
        return self.num_instances / self.batch_size

    @abstractmethod
    def generate_labels(self, config):
        """ Generate labels """
        pass

    @abstractmethod
    def collate_fn(self, data):
        pass
