from typing import Optional, Tuple
from dataclasses import dataclass

import random

from typing import Any, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

from collections import defaultdict
from torch.utils.data import Sampler, DataLoader

from romae.trainer import Trainer, TrainerConfig
import numpy as np

class PositiveGuaranteedSampler(Sampler):
    """
    Each batch contains exactly `k_positive` examples from class +1,
    and the remaining (batch_size - k_positive) slots are filled by
    randomly sampling from classes -1 and 0 (unlabeled + negatives).
    
    batch_size and k_positive are independent hyperparameters.
    """

    def __init__(self, labels, batch_size: int, k_positive: int):
        """
        Args:
            labels:     array-like with values in {-1, 0, 1}
            batch_size: total number of examples per batch
            k_positive: exact number of positives (class 1) per batch
        """
        assert k_positive < batch_size, "k_positive must be less than batch_size"
        
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.k_positive = k_positive
        self.k_rest = batch_size - k_positive

        self.positive_indices = np.where(self.labels == 1)[0]
        self.rest_indices = np.where(self.labels != 1)[0]  # -1 and 0 pooled together

        assert len(self.positive_indices) > 0, "No positive examples found"
        assert len(self.rest_indices) >= self.k_rest, "Not enough non-positive examples"

        #self.n_batches = len(self.labels) // self.batch_size
        # Epoch = one full pass over the rest pool
        self.n_batches = len(self.rest_indices) // self.k_rest
    
    def __iter__(self):
        rest_pool = np.random.permutation(self.rest_indices).tolist()
        pos_pool  = np.random.permutation(self.positive_indices).tolist()
        pos_ptr = rest_ptr = 0
    
        for _ in range(self.n_batches):
            # Rest: sequential, no wraparound needed (pool is exactly consumed)
            rest_batch = rest_pool[rest_ptr : rest_ptr + self.k_rest]
            rest_ptr += self.k_rest
    
            # Positives: wraparound/oversample as before
            pos_batch = []
            for _ in range(self.k_positive):
                if pos_ptr >= len(pos_pool):
                    pos_pool = np.random.permutation(self.positive_indices).tolist()
                    pos_ptr = 0
                pos_batch.append(pos_pool[pos_ptr])
                pos_ptr += 1
    
            batch = pos_batch + rest_batch
            np.random.shuffle(batch)
            yield batch


    def __len__(self):
        return self.n_batches

