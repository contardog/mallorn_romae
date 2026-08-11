from typing import Any, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator
import torch

torch.backends.cuda.matmul.allow_tf32 = True

import uuid

# def make_run_id() -> str:
#     return uuid.uuid4().hex[:12]

class MallornConfigContrastiveEnv(BaseSettings):
    """

    Usage:
        python3 -m romae_mallorn env_pretrain_contrastive --env_file yourenv.env

    """
    model_config = SettingsConfigDict(
        env_prefix='mallorn_',
        env_file='.env',
        extra="ignore"
    )

    # --- Data / run identity ---
    train_parquet: Optional[str] = Field(None, description="Path to training parquet")
    test_parquet: Optional[str] = Field(None, description="Path to test parquet")
    model_name: str = Field(..., description="Run name, used for checkpoint dir and W&B")

    run_id: str = Field(default_factory=lambda: __import__("uuid").uuid4().hex[:12])

    # --- Model ---
    model_size: str = Field("super-tiny")
    decoder_size: Optional[int] = Field(None, description="Decoder d_model; defaults to encoder d_model if unset")

    # --- Pretraining schedule ---
    pretrain_epochs: int = Field(400)
    pretrain_lr: float = Field(4e-4)
    pretrain_warmup_steps: int = Field(20)
    pretrain_batch_size: int = Field(128)
    pretrain_eval_every: int = Field(100)
    pretrain_save_every: int = Field(200)
    pretrain_mask_ratio: float = Field(0.5)
    pretrain_grad_clip: float = Field(1)
    max_checkpoints: int = Field(50)
    pretrain_optimargs: dict[str, Any] = {"betas": (0.9, 0.95), "weight_decay": 0.05}

    # --- Contrastive head ---
    temperature: float = Field(0.15)
    projection_head: Optional[bool] = Field(False)
    projection_hidden_dim: Optional[int] = Field(None)  ## This is not used with the new onelayer projection head
    cls_contrastive_dim: Optional[int] = Field(None)
    aug_contrast_weight: float = Field(0.0) ## By default we don't do that
    class_contrast_weight: float = Field(1.0)
    recon_weight: float = Field(1.0)
    n_views: int = Field(2)
    mask_ratio_contrastive: float = Field(0.5)
    decode: bool = Field(True)
    
    unsup_in_denominator: bool = Field(False)

    # --- Sampler ---
    K_positive_batch: Optional[int] = Field(
        None,
        description="If set, forces exactly this many positives per minibatch"
    )
    K_negative_batch: Optional[int] = Field(
        None,
        description="If set, forces exactly this many negatives per minibatch")

    # N_positive_dataset: Optional[int] = Field(
    #     None,
    #     description='If set, creates a subsampled training/val datasets from the parquet file given... train parquet? Ignore Test parquet?'
    # )

    
    # --- Resampling ---
    train_pool_parquet: Optional[str] = Field(
        None, description="Full pool parquet to resample train/val from. If set, overrides train_parquet/test_parquet as data source."
    )
    id_column: str = Field("ObjectId", description="Column in pool parquet giving a stable unique identifier per object")
    n_negative: Optional[int] = Field(None, description="Number of negatives to sample into training set")
    n_unsup: Optional[int] = Field(None, description="Number of unlabeled examples to sample into training set")
    n_positive: Optional[int] = Field(None, description="Number of positive examples to sample into training set")
    val_fraction: float = Field(0.2, description="Fraction of each class held out for validation")
    resample_seed: Optional[int] = Field(None, description="Seed for train/val resampling; if None, drawn randomly and logged")
    resample_with_replacement: bool = Field(False) ## Probably should force this to False always because of some issues in sample_train_vla
    
    # --- Sampler, extended ---
    K_unlabeled_batch: Optional[int] = Field(
        None, description="If set, forces exactly this many unlabeled examples per minibatch"
    )
    n_batches_per_epoch: Optional[int] = Field(
        None, description="Fixes batches/epoch. Should be computed once off the LARGEST N_positive_dataset in a sweep and reused for every run in that sweep, so all runs get equal optimizer step budgets."
    )

    
        
    ### Need something to save the file for idx then? Or make it with a randomly generated tag that has to match the model otherwise it's going 
    ### to be a fucking mess
    ### file_idx_trainingsample: Optional[int] = Field( None, description='Required if N_positive_dataset is set')
    
    # --- Dataset / Observation dropout stuff --- 
    
    obs_dropout_end_trim: float = Field(0.05)      # fraction of seq to trim from ends
    obs_dropout_edge_erosion: float = Field(0.04)  # max fraction to erode per gap edge
    gap_threshold_factor: float = Field(20.0)       # median_dt * this = gap threshold
    random_dropout_ratio: float = Field(0.04)

    # --- Misc ---
    gaussian_noise: bool = Field(False)
    vega: bool = Field(False, description="Set True on cluster to use sched_getaffinity for worker count")
    project_name: str = Field("contrastive_")
    entity_name: str = Field("contardog-university-of-nova-gorica")


    @model_validator(mode="after")
    def _check_data_source(self):
        if self.train_pool_parquet is None and (self.train_parquet is None or self.test_parquet is None):
            raise ValueError(
                "Set either `train_pool_parquet` (resample train/val from a pool), "
                "or both `train_parquet` and `test_parquet` (use pre-split files directly)."
            )
        return self