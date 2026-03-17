"""
Contrastive Pre-training Script for RoMAE

Uses RoMAEPreTrainingContrastive which extends RoMAEForPreTraining.
"""
from romae.model import _get_attn_mask
import torch
from torch.utils.data import DataLoader
from romae.model import RoMAEForPreTrainingConfig

import torch.nn as nn

import pandas as pd
import numpy as np
from romae.model import RoMAEForPreTrainingConfig, EncoderConfig, RoMAEBase, Encoder

from romae_mallorn.romae_contrastive import RoMAEPreTrainingContrastive, RandomMasking

from romae_mallorn.dataset import MallornDataset, MallornDatasetwLabel, gen_mask, padd_parquet
from romae_mallorn.config import MallornConfig, MallornConfigContrastive
import polars as pl
from romae.trainer import Trainer, TrainerConfig

from romae.utils import get_drop_path, patchify, load_from_checkpoint, get_encoder_size

import os

import tqdm

from romae_mallorn.utils import override_encoder_size

import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass

import random

from typing import Any, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field



# import logging

# logger = logging.getLogger(__name__)

from romae.utils import get_encoder_size
from romae.model import RoMAEForPreTraining, RoMAEForPreTrainingConfig, EncoderConfig
from romae.trainer import Trainer, TrainerConfig

from romae_mallorn.dataset import MallornDataset
from romae_mallorn.config import MallornConfig
from romae_mallorn.utils import override_encoder_size

from collections import defaultdict
from torch.utils.data import Sampler, DataLoader

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


class TrainerConfigSampler(TrainerConfig):

    K_positive_batch: Optional[int] = Field(
        None,
        description="Number of positive examples guaranteed per minibatch"
    )
    
    
class TrainerSampler(Trainer):
    
    def __init__(self, config: TrainerConfig):
        super().__init__(config)
        

    def get_dataloaders(self, train_dataset, test_dataset,
                            train_collate_fn, eval_collate_fn):

        if self.config.K_positive_batch is not None:   
            print("OVERRIDING DATA LOADERS FROM TRAINER")
            print(f"K positive examples {self.config.K_positive_batch}")
            
            sampler_train = PositiveGuaranteedSampler(
                labels=train_dataset.get_labels(),   # array of {-1, 0, 1}
                batch_size=self.config.batch_size,            
                k_positive=self.config.K_positive_batch,             
            )
    
            sampler_test = PositiveGuaranteedSampler(
                labels=test_dataset.get_labels(),   # array of {-1, 0, 1}
                batch_size=self.config.batch_size,            
                k_positive=self.config.K_positive_batch,             
            )
    
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                #batch_size=self.config.batch_size,
                num_workers=self.config.num_dataset_workers,
                pin_memory=True,
                #shuffle=True,
                batch_sampler = sampler_train,
                collate_fn=train_collate_fn,
                prefetch_factor=2,
            )
            
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                #batch_size=self.config.batch_size,
                num_workers=self.config.num_dataset_workers,
                pin_memory=True,
                batch_sampler = sampler_test, ## bit shady to have it in test as well but it ensures the loss is the same?!
                collate_fn=eval_collate_fn,
                prefetch_factor=2
            )
        else:
            train_dataloader, test_dataloader = super().get_dataloaders(train_dataset, test_dataset,
                            train_collate_fn, eval_collate_fn)
            
        return train_dataloader, test_dataloader

def pretrain_contrastive(args):
    """
    Pre-training script which will train RoMAForPreTraining on the data.
    """
    
    contrastive_config = MallornConfigContrastive(
        # temperature=args.get('temperature', 0.15),
        # projection_dim=args.get('projection_dim', 64),
        # projection_hidden_dim=args.get('projection_hidden_dim', 64),
        # cls_contrastive_dim=args.get('cls_contrastive_dim', None),
        # aug_contrast_weight=args.get('aug_contrast_weight', 1.0),
        # class_contrast_weight=args.get('class_contrast_weight', 1.0),
        # recon_weight=args.get('recon_weight', 1.0),
        # mask_ratio=config.get('mask_ratio_contrastive', 0.5)
    )
    
    if args.temperature is not None:
        print("Overridding configured temperature")
        contrastive_config.temperature = args.temperature
    if args.projection_dim is not None:
        print("Overridding configured projection_dim")
        contrastive_config.projection_dim = args.projection_dim
    if args.cls_contrastive_dim is not None:
        print("Overridding configured cls_contrastive_dim")
        contrastive_config.cls_contrastive_dim = args.cls_contrastive_dim
    if args.augm_contrast_weight is not None:
        print("Overridding configured aug_contrast_weight")
        contrastive_config.aug_contrast_weight = args.augm_contrast_weight
    if args.class_contrast_weight is not None:
        print("Overridding configured class_contrast_weight")
        contrastive_config.class_contrast_weight = args.class_contrast_weight
    if args.class_contrast_weight is not None:
        print("Overridding configured class_contrast_weight")
        contrastive_config.class_contrast_weight = args.class_contrast_weight
    if args.recon_weight is not None:
        print("Overridding configured recon_weight")
        contrastive_config.recon_weight = args.recon_weight
    if args.contrastive_mask_ratio is not None:
        print("Overridding configured mask_ratio_contrastive")
        contrastive_config.mask_ratio_contrastive = args.contrastive_mask_ratio


    if args.no_decode:
        contrastive_config.decode = False
        
    #temperature: float = 0.15
    #projection_dim: int = 32
    #cls_contrastive_dim: Optional[int] = 32  # Split CLS token if set
    #aug_contrast_weight: float = 1.0
    #class_contrast_weight: float = 1.0
    #recon_weight: float = 0.0
    #mask_ratio_contrastive = 0.5
    

    ## Make it smaller?! Add some piece of code for this
    if args.model_size is not None:
        print("Overriding configured model size!!!!")
        contrastive_config.model_size = args.model_size
    
    encoder_args = override_encoder_size(contrastive_config.model_size)

    
    ## one could argue that the else should be some factor of encoder_args dim? Eh.
    
    decoder_size = args.decoder_size if args.decoder_size is not None else encoder_args['d_model']
    
    ## Forcing the same size of embeddings in decoder but a shallow depth
    decoder_args = {
                    "d_model": decoder_size,
                    "nhead": 3,
                    "depth": 2 ## I guess this is arbitrary? We could have it at = 1? 
                }
    
    
    model_config = RoMAEForPreTrainingConfig(
        encoder_config=EncoderConfig(**encoder_args),
        decoder_config= EncoderConfig(**decoder_args),
        tubelet_size=(1, 1, 1),
        n_channels=2,
        n_pos_dims=2
    )

    # if args.no_cls:
    #     model_config.use_cls = False
        
    print(model_config)
    
    if args.lr is not None:
        print("Overridding configured learning rate")
        contrastive_config.pretrain_lr = args.lr
    if args.batch_size is not None:
        print("Overriding configured batch size")
        contrastive_config.pretrain_batch_size = args.batch_size
        contrastive_config.eval_batch_size = args.batch_size
    if args.epochs is not None:
        print("Overridding configured number of epochs")
        contrastive_config.pretrain_epochs = args.epochs

    if args.pretrain_mask_ratio is not None:
        print("Overriding mask ratio")
        contrastive_config.pretrain_mask_ratio = args.pretrain_mask_ratio

    print("Contrastive config")
    print(contrastive_config)
    
    # Create augmentation
    augmentation = RandomMasking(contrastive_config.mask_ratio_contrastive) 
    ## Might mean we get only 25% of the data if we have two masking steps...
    
    # Create model (this is a subclass of RoMAEForPreTraining)
    model = RoMAEPreTrainingContrastive(
        config=model_config,
        contrastive_config=contrastive_config,
        augmentation_fn=augmentation
    ) #.to(device)

    

    #if args.K_pos_batch is not None:    
    trainer_config = TrainerConfigSampler(
        warmup_steps=contrastive_config.pretrain_warmup_steps,
        checkpoint_dir=args.model_name+"_checkpoint_",
        epochs=contrastive_config.pretrain_epochs,
        base_lr=contrastive_config.pretrain_lr,
        eval_every=contrastive_config.pretrain_eval_every,
        save_every=contrastive_config.pretrain_save_every,
        optimizer_args=contrastive_config.pretrain_optimargs,
        batch_size= contrastive_config.pretrain_batch_size,
        project_name= contrastive_config.project_name + args.model_name,
        entity_name='contardog-university-of-nova-gorica',
        gradient_clip=contrastive_config.pretrain_grad_clip,
        lr_scaling=True,
        K_positive_batch = args.K_pos_batch
        #max_checkpoints = 20,
    )
    # else:         
    #     trainer_config = TrainerConfig(
    #         warmup_steps=contrastive_config.pretrain_warmup_steps,
    #         checkpoint_dir=args.model_name+"_checkpoint_",
    #         epochs=contrastive_config.pretrain_epochs,
    #         base_lr=contrastive_config.pretrain_lr,
    #         eval_every=contrastive_config.pretrain_eval_every,
    #         save_every=contrastive_config.pretrain_save_every,
    #         optimizer_args=contrastive_config.pretrain_optimargs,
    #         batch_size= contrastive_config.pretrain_batch_size,
    #         project_name= contrastive_config.project_name + args.model_name,
    #         entity_name='contardog-university-of-nova-gorica',
    #         gradient_clip=contrastive_config.pretrain_grad_clip,
    #         lr_scaling=True
    #         #max_checkpoints = 20,
    #     )
    
    if (args.vega):
        print("Original trainer config num_dataset_workerS")
        print(trainer_config.num_dataset_workers)
        print("Overriding")
        trainer_config.num_dataset_workers=len(os.sched_getaffinity(0)) #*2
        print(trainer_config.num_dataset_workers)
        
    print("Start pretrain")
    
    trainer = TrainerSampler(trainer_config)
    with (
        MallornDatasetwLabel(args.test_parquet, mask_ratio=contrastive_config.pretrain_mask_ratio) as test_dataset,
        MallornDatasetwLabel(args.train_parquet,                
                         gaussian_noise=contrastive_config.gaussian_noise, mask_ratio=contrastive_config.pretrain_mask_ratio) as train_dataset
    ):
        trainer.train(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
        )

