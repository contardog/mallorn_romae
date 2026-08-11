"""
Contrastive Pre-training Script — env-file driven version.

All hyperparameters come from a .env file (or inline env vars).
No argparse overrides. See MallornConfigContrastiveEnv for all fields.

Example:
    cp experiments/verytiny10_20dim.env .env
    python3 -m romae_mallorn env_pretrain_contrastive

    # or one-off override inline:
    MALLORN_TEMPERATURE=0.3 python3 -m romae_mallorn env_pretrain_contrastive

    # or point to a specific env file directly:
    python3 -m romae_mallorn env_pretrain_contrastive --env_file experiments/verytiny10_20dim.env
"""

import os
import numpy as np
import torch

import torch.nn as nn
from typing import Optional
from collections import defaultdict

from pydantic import Field
from torch.utils.data import Sampler

from romae.model import RoMAEForPreTrainingConfig, EncoderConfig
from romae.trainer import Trainer, TrainerConfig

from romae_mallorn.romae_contrastive import RoMAEPreTrainingContrastive, RandomMasking
from romae_mallorn.dataset import MallornDatasetwLabelTrimMask
from romae_mallorn.utils import override_encoder_size
from romae_mallorn.env_config import MallornConfigContrastiveEnv
from romae_mallorn.samplers import append_run_manifest


from romae_mallorn.samplers import PositiveGuaranteedSampler, save_index_record, sample_train_val
from romae_mallorn.trainers import TrainerSampler, TrainerConfigSampler
import shutil
import uuid
import polars as pl

import wandb

def make_run_id() -> str:
    return uuid.uuid4().hex[:12]
    
def env_pretrain_contrastive(args):
    """
    Pretrain with config driven entirely by env file.
    args only carries --env_file (optional), nothing else.
    """
    # Load config — either from explicit --env_file or default .env
    config = MallornConfigContrastiveEnv(
        _env_file=args.env_file if args.env_file else '.env'
    )
    # run_id = make_run_id()
    # config.run_id = run_id 
    
    # Save the env file alongside the checkpoints 
    ## this will allow to reconstruct this exact run as python3 -m romae_mallorn env_pretrain_contrastive --env_file checkpointdir_checkpoint_/config.env

    #os.makedirs(config.model_name + "_checkpoint_", exist_ok=True)
    ## TODO EDIT MODEL NAME TO HAVE RUN ID 

    
    run_id = config.run_id
    index_path = f"{config.model_name}_{run_id}_indices.json"
    
    if config.train_pool_parquet is not None:
        pool = pl.read_parquet(config.train_pool_parquet)
        seed = config.resample_seed if config.resample_seed is not None else np.random.randint(0, 2**31 - 1)
        config.resample_seed = seed
        train_df, val_df, record = sample_train_val(
            pool=pool,
            id_column=config.id_column,
            n_positive=config.n_positive,
            n_negative=config.n_negative,
            n_unsup=config.n_unsup,
            val_fraction=config.val_fraction,
            seed=seed,
            with_replacement=config.resample_with_replacement,
        )
        save_index_record(record, index_path)
        # pass train_df / val_df directly into the Dataset, see (5) below
    elif config.train_parquet is not None:
        train_df = config.train_parquet
        val_df = config.test_parquet
    else:
        print("This should have broken earlier")
        return 0
        
    
    append_run_manifest(config)  # writes one line to runs_manifest.jsonl
    
    # still keep a per-run copy of the env file, but keyed by run_id, not model_name,
    # so it can never collide with a previous run of the same model_name
    shutil.copy(
        args.env_file if args.env_file else '.env',
        f"{config.model_name}_{config.run_id}_config.env"
    )

    
    print("=== MallornConfigContrastiveEnv ===")
    print(config.model_dump_json(indent=2))

    # --- Model architecture ---
    encoder_args = override_encoder_size(config.model_size)
    encoder_args["drop_path_rate"]=0.25
    encoder_args["hidden_drop_rate"]=0.15
    encoder_args["pos_drop_rate"]=0.1
    encoder_args["attn_drop_rate"]=0.1
    encoder_args["attn_proj_drop_rate"]=0.1
    
    decoder_size = config.decoder_size or encoder_args['d_model']
    decoder_args = {
        "d_model": decoder_size,
        "nhead": 3,
        "depth": 2,
        "drop_path_rate": 0.05
    }

    model_config = RoMAEForPreTrainingConfig(
        encoder_config=EncoderConfig(**encoder_args),
        decoder_config=EncoderConfig(**decoder_args),
        tubelet_size=(1, 1, 1),
        n_channels=2,
        n_pos_dims=2,
    )
    print(model_config)


    # contrastive_config = MallornConfigContrastive()
    # contrastive_config.temperature = config.temperature
    # contrastive_config.projection_dim = config.projection_dim
    # contrastive_config.projection_hidden_dim = config.projection_hidden_dim
    # contrastive_config.cls_contrastive_dim = config.cls_contrastive_dim
    # contrastive_config.aug_contrast_weight = config.aug_contrast_weight
    # contrastive_config.class_contrast_weight = config.class_contrast_weight
    # contrastive_config.recon_weight = config.recon_weight
    # contrastive_config.mask_ratio_contrastive = config.mask_ratio_contrastive
    # contrastive_config.decode = config.decode
    # contrastive_config.gaussian_noise = config.gaussian_noise
    # contrastive_config.pretrain_mask_ratio = config.pretrain_mask_ratio

    augmentation = RandomMasking(config.mask_ratio_contrastive)

    model = RoMAEPreTrainingContrastive(
        config=model_config,
        contrastive_config=config,
        augmentation_fn=augmentation,
    )

    ## probably need to have this in config
    print("UPDATE CONFIG< SWITCH MSE FOR SMOOTH L1")
    model.set_loss_fn(nn.HuberLoss(reduction='none', delta=0.5)) # delta: need to check the actual flux distrib now that we're rescaling?

    
    log_dir = config.model_name  # or however you read it
    print(f"Log dir is:{log_dir}")
    #project_name_wandb = f"Mallorn_{os.path.basename(log_dir)}"

    if config.train_pool_parquet is not None:
        dataset_tag = os.path.basename(config.train_pool_parquet)
    else:
        dataset_tag = os.path.basename(config.train_parquet)
    
    project_name_wandb = f"Mallorn_{dataset_tag}"
    
    
    # e.g. "Mallorn_ELAsTiCC2_150pos_3000neg"
    print(f"Project name is {project_name_wandb}")

    #run_name = f"{config.model_size}_recon{config.recon_weight}_supcon{config.class_contrast_weight}_temp{config.temperature}"
    n_pos_tag = config.n_positive if config.n_positive is not None else "full"
    run_name = (
        f"{config.model_size}_recon{config.recon_weight}_supcon{config.class_contrast_weight}"
        f"_temp{config.temperature}_Npos{n_pos_tag}_seed{seed if config.train_pool_parquet else 'na'}"
        f"_{config.run_id[:6]}"
    )

    print(f"run name is {run_name}")
    
    # --- Trainer ---
    trainer_config = TrainerConfigSampler(
        warmup_steps=config.pretrain_warmup_steps,
        checkpoint_dir=config.model_name + "_checkpoint_",
        epochs=config.pretrain_epochs,
        base_lr=config.pretrain_lr,
        eval_every=config.pretrain_eval_every,
        save_every=config.pretrain_save_every,
        optimizer_args=config.pretrain_optimargs,
        batch_size=config.pretrain_batch_size,
        project_name=config.project_name + project_name_wandb,
        entity_name=config.entity_name,
        run_name=run_name,
        gradient_clip=config.pretrain_grad_clip,
        lr_scaling=True,
        K_positive_batch=config.K_positive_batch,   
        K_negative_batch=config.K_negative_batch,
        K_unlabeled_batch=config.K_unlabeled_batch,
        n_batches_per_epoch=config.n_batches_per_epoch,
    )

    if config.vega:
        print(f"[vega] Overriding num_dataset_workers to {len(os.sched_getaffinity(0))}")
        trainer_config.num_dataset_workers = len(os.sched_getaffinity(0))

    print("Starting contrastive pretraining...")
    trainer = TrainerSampler(trainer_config)

    
    if (trainer.run is not None) and (config.train_pool_parquet is not None):
        trainer.run.config.update({"run_id": config.run_id, "index_path": index_path})
        artifact = wandb.Artifact(f"indices-{config.run_id}", type="train_val_indices")
        artifact.add_file(index_path)
        artifact.add_file(f"{config.model_name}_{config.run_id}_config.env")
        trainer.run.log_artifact(artifact)

    if config.n_batches_per_epoch is None:
        print(f"[GuaranteedQuotaSampler] n_batches_per_epoch not set — deriving locally per-run "
          f"(epoch length will scale with pool size; fine for a single run, "
          f"NOT fine for an N-sweep you want compute-matched)")

    with (
        MallornDatasetwLabelTrimMask(val_df,
                             mask_ratio=config.pretrain_mask_ratio,                              
                            ### Config for observation dropout in Dataset -- not sure we want it in test??
                            obs_dropout_end_trim = config.obs_dropout_end_trim ,      # fraction of seq to trim from ends
                            obs_dropout_edge_erosion = config.obs_dropout_edge_erosion,   # max fraction to erode per gap edge
                            gap_threshold_factor = config.gap_threshold_factor,      # mjd
                            random_dropout_ratio = config.random_dropout_ratio,
                                     training = False
                                    ) as test_dataset,
        MallornDatasetwLabelTrimMask(train_df,
                             gaussian_noise=config.gaussian_noise,                         
                            ### Config for observation dropout in Dataset
                             mask_ratio=config.pretrain_mask_ratio,
                            obs_dropout_end_trim = config.obs_dropout_end_trim ,      # fraction of seq to trim from ends
                            obs_dropout_edge_erosion = config.obs_dropout_edge_erosion,   # max fraction to erode per gap edge
                            gap_threshold_factor = config.gap_threshold_factor,      # mjd
                            random_dropout_ratio = config.random_dropout_ratio) as train_dataset,
    ):
        trainer.train(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
        )
