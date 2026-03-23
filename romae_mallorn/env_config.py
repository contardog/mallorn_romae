from typing import Any, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import torch

torch.backends.cuda.matmul.allow_tf32 = True


class MallornConfigContrastiveEnv(BaseSettings):
    """
    Drop-in replacement for MallornConfigContrastive + argparse overrides.
    Everything that was previously a CLI arg is now a field here,
    readable from a .env file or inline env vars.

    Usage:
        cp my_experiment.env .env && python3 -m romae_mallorn env_pretrain_contrastive
        or
        python3 -m romae_mallorn env_pretrain_contrastive --env_file yourenv.env

        # or inline:
        MALLORN_TEMPERATURE=0.3 python3 -m romae_mallorn env_pretrain_contrastive
    """
    model_config = SettingsConfigDict(
        env_prefix='mallorn_',
        env_file='.env',
        extra="ignore"
    )

    # --- Data / run identity ---
    train_parquet: str = Field(..., description="Path to training parquet")
    test_parquet: str = Field(..., description="Path to test parquet")
    model_name: str = Field(..., description="Run name, used for checkpoint dir and W&B")

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
    projection_dim: Optional[int] = Field(None) ## In the new SupCon version this will not be used (rely on cls_contrastive_dim) because no projection head
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

    # --- Dataset / Observation dropout stuff --- 
    
    obs_dropout_end_trim: float = Field(0.05)      # fraction of seq to trim from ends
    obs_dropout_edge_erosion: float = Field(0.04)  # max fraction to erode per gap edge
    gap_threshold_factor: float = Field(20.0)       # median_dt * this = gap threshold
    random_dropout_ratio: float = Field(0.04)

    # --- Misc ---
    gaussian_noise: bool = Field(False)
    vega: bool = Field(False, description="Set True on cluster to use sched_getaffinity for worker count")
    project_name: str = Field("Mallorn_contrastive_")
    entity_name: str = Field("contardog-university-of-nova-gorica")