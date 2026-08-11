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



import json
from pathlib import Path
import polars as pl


from datetime import datetime, timezone

def append_run_manifest(config, manifest_path: str = "runs_manifest.jsonl"):
    """
    Appends one JSON line per run with the full config + run_id + timestamp.
    jsonl (not a single JSON/parquet file) so concurrent/parallel runs never
    clobber each other on write -- just append.
    """
    record = config.model_dump()
    record["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with open(manifest_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
        
"""
Train/val resampling from a full pool parquet, with saved, run-tagged indices.
"""
def sample_train_val(
    pool: pl.DataFrame,
    id_column: str,
    n_positive: int,
    n_negative: Optional[int],
    n_unsup: Optional[int],
    val_fraction: float,
    seed: int,
    with_replacement: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """
    Draws n_positive/n_negative/n_unsup examples from `pool` (labels assumed
    in a 'binary_class' column with values {1, 0, -1}), splits each class
    into train/val by val_fraction, and returns (train_df, val_df, index_record).

    index_record is a JSON-serializable dict capturing exactly which IDs went
    where, so it can be saved and later reloaded for the secondary classifier
    stage without leakage.
    """
    rng = np.random.default_rng(seed)
    record = {"seed": seed, "id_column": id_column, "with_replacement": with_replacement}

    class_specs = [(1, n_positive), (0, n_negative), (-1, n_unsup)]
    train_frames, val_frames = [], []

    for label, n in class_specs:
        if n is None:
            continue
        sub = pool.filter(pl.col("binary_class") == label)
        ids = sub[id_column].to_numpy()
        if len(ids) == 0:
            raise ValueError(f"No examples with binary_class == {label} in pool")

        if with_replacement:
            chosen = rng.choice(ids, size=n, replace=True)
        else:
            if n > len(ids):
                raise ValueError(f"Requested {n} for class {label} but pool only has {len(ids)}")
            chosen = rng.choice(ids, size=n, replace=False)

        chosen = np.unique(chosen) if not with_replacement else chosen
        n_val = max(1, int(len(chosen) * val_fraction))
        # shuffle chosen (np.unique sorts) before splitting train/val
        perm = rng.permutation(len(chosen))
        chosen = chosen[perm]
        val_ids, train_ids = chosen[:n_val], chosen[n_val:]

        record[f"class_{label}_train_ids"] = train_ids.tolist()
        record[f"class_{label}_val_ids"] = val_ids.tolist()

        train_frames.append(sub.filter(pl.col(id_column).is_in(train_ids.tolist())))
        val_frames.append(sub.filter(pl.col(id_column).is_in(val_ids.tolist())))

    train_df = pl.concat(train_frames)
    val_df = pl.concat(val_frames)
    return train_df, val_df, record


def save_index_record(record: dict, path: str):
    Path(path).write_text(json.dumps(record, indent=2))


def load_index_record(path: str) -> dict:
    return json.loads(Path(path).read_text())


def filter_pool_by_record(pool: pl.DataFrame, record: dict, id_column: str, split: str) -> pl.DataFrame:
    """split: 'train' or 'val'. Reconstructs the exact same subset from the pool
    given a previously-saved record, for the secondary classifier stage."""
    ids = []
    for key, val in record.items():
        if key.endswith(f"_{split}_ids"):
            ids.extend(val)
    return pool.filter(pl.col(id_column).is_in(ids))

class GuaranteedQuotaSampler(Sampler):
    def __init__(self, labels, k_positive: int, k_negative: int,
                 k_unlabeled: int = 0, n_batches: Optional[int] = None):
        self.labels = np.array(labels)
        self.k_positive = k_positive
        self.k_negative = k_negative
        self.k_unlabeled = k_unlabeled

        self.positive_indices = np.where(self.labels == 1)[0]
        self.negative_indices = np.where(self.labels == 0)[0]
        self.unlabeled_indices = np.where(self.labels == -1)[0]

        assert len(self.positive_indices) > 0, "No positive examples found"
        if k_negative > 0:
            assert len(self.negative_indices) > 0, "No negative examples found"
        if k_unlabeled > 0:
            assert len(self.unlabeled_indices) > 0, "No unlabeled examples found"

        if n_batches is not None:
            self.n_batches = n_batches
        else:
            # fallback: derive locally so every configured pool is fully covered
            # at least once this epoch (same semantics as before, just per-run).
            candidates = [int(np.ceil(len(self.positive_indices) / k_positive))]
            if k_negative > 0:
                candidates.append(int(np.ceil(len(self.negative_indices) / k_negative)))
            if k_unlabeled > 0:
                candidates.append(int(np.ceil(len(self.unlabeled_indices) / k_unlabeled)))
            self.n_batches = max(candidates)

    # __iter__ / __len__ unchanged
    @staticmethod
    def _draw(pool: np.ndarray, k: int, ptr: int, shuffled: list) -> tuple[list, int, list]:
        batch = []
        for _ in range(k):
            if ptr >= len(shuffled):
                shuffled = np.random.permutation(pool).tolist()
                ptr = 0
            batch.append(shuffled[ptr])
            ptr += 1
        return batch, ptr, shuffled

    def __iter__(self):
        pos_shuf = np.random.permutation(self.positive_indices).tolist()
        neg_shuf = np.random.permutation(self.negative_indices).tolist() if self.k_negative > 0 else []
        unsup_shuf = np.random.permutation(self.unlabeled_indices).tolist() if self.k_unlabeled > 0 else []
        pos_ptr = neg_ptr = unsup_ptr = 0

        for _ in range(self.n_batches):
            pos_batch, pos_ptr, pos_shuf = self._draw(self.positive_indices, self.k_positive, pos_ptr, pos_shuf)
            batch = list(pos_batch)
            if self.k_negative > 0:
                neg_batch, neg_ptr, neg_shuf = self._draw(self.negative_indices, self.k_negative, neg_ptr, neg_shuf)
                batch += neg_batch
            if self.k_unlabeled > 0:
                unsup_batch, unsup_ptr, unsup_shuf = self._draw(self.unlabeled_indices, self.k_unlabeled, unsup_ptr, unsup_shuf)
                batch += unsup_batch
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches
    
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

