from typing import Optional, Tuple
from dataclasses import dataclass
import torch
import random
import json
import tqdm

from typing import Any, Optional

from pathlib import Path
from datetime import datetime

import wandb
from accelerate import Accelerator

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

from collections import defaultdict
from torch.utils.data import Sampler, DataLoader

from romae.trainer import Trainer, TrainerConfig

from romae_mallorn.samplers import PositiveGuaranteedSampler, GuaranteedQuotaSampler

class TrainerConfigSampler(TrainerConfig):

    K_positive_batch: Optional[int] = Field(None, description="Positives guaranteed per minibatch")
    K_negative_batch: Optional[int] = Field(None, description="Negatives guaranteed per minibatch; derived from batch_size - K_positive - K_unlabeled if unset")
    K_unlabeled_batch: Optional[int] = Field(None, description="Unlabeled examples guaranteed per minibatch")
    n_batches_per_epoch: Optional[int] = Field(None, description="Fixed batches/epoch across an N-sweep")

    
    
class TrainerSampler(Trainer):
    
    def __init__(self, config: TrainerConfig, save_best_val=True):
        super().__init__(config)
        self.val_loss = None
        self.save_best_val = save_best_val # move into config
        self.val_checkpoint_dir = self.config.checkpoint_dir + "val"
        
    
        
    def _build_quota_sampler(self, dataset, n_batches_override=None):
        k_positive = self.config.K_positive_batch
        k_unlabeled = self.config.K_unlabeled_batch or 0
        k_negative = (
            self.config.K_negative_batch
            if self.config.K_negative_batch is not None
            else self.config.batch_size - k_positive - k_unlabeled
        )
        assert k_positive + k_negative + k_unlabeled == self.config.batch_size, (
            f"k_positive({k_positive}) + k_negative({k_negative}) + k_unlabeled({k_unlabeled}) "
            f"!= batch_size({self.config.batch_size})"
        )
        return GuaranteedQuotaSampler(
            labels=dataset.get_labels(),
            k_positive=k_positive,
            k_negative=k_negative,
            k_unlabeled=k_unlabeled,
            n_batches=n_batches_override,
        )

    def get_dataloaders(self, train_dataset, test_dataset, train_collate_fn, eval_collate_fn):
        if self.config.K_positive_batch is not None:
            print("OVERRIDING DATA LOADERS FROM TRAINER")
            print(f"K positive={self.config.K_positive_batch}, "
                  f"K negative={self.config.K_negative_batch}, "
                  f"K unlabeled={self.config.K_unlabeled_batch}, "
                  f"n_batches={self.config.n_batches_per_epoch}")

            sampler_train = self._build_quota_sampler(train_dataset, n_batches_override=self.config.n_batches_per_epoch)
        
            n_val_positive = int((test_dataset.get_labels() == 1).sum())
            n_batches_eval = -(-n_val_positive // self.config.K_positive_batch)  # ceil division
            sampler_test = self._build_quota_sampler(test_dataset, n_batches_override=n_batches_eval)


            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                num_workers=self.config.num_dataset_workers,
                pin_memory=True,
                batch_sampler=sampler_train,
                collate_fn=train_collate_fn,
                prefetch_factor=2,
            )
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                num_workers=self.config.num_dataset_workers,
                pin_memory=True,
                batch_sampler=sampler_test,
                collate_fn=eval_collate_fn,
                prefetch_factor=2,
            )
        else:
            train_dataloader, test_dataloader = super().get_dataloaders(
                train_dataset, test_dataset, train_collate_fn, eval_collate_fn
            )
        return train_dataloader, test_dataloader


    def save_checkpoint_valid(self, accelerator: Accelerator, step_counter,
                        model_config):
        if self.val_loss is None:            
            Path(self.val_checkpoint_dir).mkdir(exist_ok=True, parents=True)

        savedir = Path(self.val_checkpoint_dir)/f"{step_counter}" ## switch if we want to save the N-best val, otherwise only one rn
        savedir.mkdir(exist_ok=True)

        accelerator.save_state(str(savedir))

        with open(savedir/"model_config.json", "w") as f:
            json.dump(model_config.model_dump(), f, indent=4)

        with open(savedir/"trainer_config.json", "w") as f:
            json.dump(self.config.model_dump(), f, indent=4)

        with open(savedir/"trainer_state.json", "w") as f:
            json.dump({"step": step_counter}, f, indent=4)

        #self.remove_old_checkpoints(self.config.checkpoint_dir)
        
    def evaluate(self, accelerator, model, loss_train, test_dataloader, optim, step):
        if loss_train is None:
            return
    
        grads = [
            param.grad.detach().flatten()
            for param in model.parameters()
            if param.grad is not None
        ]
        if self.run is not None:
            norm = torch.cat(grads).norm()
            self.run.log({"train/gradient_norm": norm}, step=step)
            self.run.log({"train/lr": optim.param_groups[0]["lr"]}, step=step)
            if loss_train is not None:
                self.run.log({"loss/train": loss_train.item()}, step=step)
                print(f"Train loss: {loss_train.item()}\n")
        loss = 0
        for modelargs in tqdm.tqdm(test_dataloader, desc="Evaluating"):
            modelargs = {key: val.to(accelerator.device) for key, val in
                         modelargs.items()}
            _, loss_ = model(**modelargs)
            loss = loss + loss_ / len(test_dataloader)
        if self.run is not None:
            self.run.log({"loss/validation": loss.item()}, step=step)

        if self.save_best_val and (self.val_loss is None or loss.item() < self.val_loss):
            print("Saving new best validation model")
            with torch.no_grad():
                self.save_checkpoint_valid(accelerator, "_bestval", model.config)
            self.val_loss = loss.item()
                    
        self.evaluate_callback(model, loss_train, test_dataloader)
