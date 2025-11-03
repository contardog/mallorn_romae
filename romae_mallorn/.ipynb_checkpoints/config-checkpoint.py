from typing import Any, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import torch

# Allow utilization of tensor cores
torch.backends.cuda.matmul.allow_tf32 = True

class MallornConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='mallorn_',
        env_file='.env',
        extra="ignore"
    )
    pretrained_model: Optional[str] = Field(None)
    eval_checkpoint: Optional[str] = Field(None)
    eval_batch_size: Optional[int] = Field(128)
    gaussian_noise: Optional[bool] = Field(False)
    model_size: str = Field("super-tiny")
    
    pretrain_epochs: int = Field(400)
    pretrain_lr: float = Field(4e-4)
    pretrain_warmup_steps: int = 20
    pretrain_batch_size: int = Field(128)
    pretrain_eval_every: int = Field(200) ## In number of batches
    pretrain_save_every: int = Field(200) # In number of batches -- it will retain up to N = max_checkpoints i think
    pretrain_mask_ratio: float = Field(0.4)
    pretrain_grad_clip: float = Field(10)
    max_checkpoints: int = Field(50)
    
    ## 
    
    finetune_epochs: int = Field(25)
    finetune_lr: float = Field(1e-4)
    #? Check warmup
    finetune_warmup_steps: int = 0 # 2000
    finetune_batch_size: int = Field(128)
    finetune_eval_every: int = Field(200)
    finetune_save_every: int = Field(200)
    finetune_grad_clip: float = Field(10)
    finetune_label_smoothing: float = Field(0.0) #?
    
    finetune_use_class_weights: bool = Field(True)
    class_weights: list[float] = Field([0.1, 10]) # Trying something

    ## Example from elasticc:
    # class_weights: list[float] = Field([
    #     0.2011, 1.2752, 0.7855, 0.6691, 1.6178, 0.2065, 1.4077, 2.4635, 7.7423,
    #     0.1898, 0.9508, 0.1902, 0.3751, 0.0944, 0.3725, 0.0644, 0.0449, 0.1929,
    #     0.1939, 0.7324
    # ])

    
    n_classes: int = Field(2)
    
    
    project_name: str = Field("Mallorn_")
    finetune_optimargs: dict[str, Any] = {"betas": (0.9, 0.999),
                                           "weight_decay": 0.05}
    pretrain_optimargs: dict[str, Any] = {"betas": (0.9, 0.95),
                                          "weight_decay": 0.05}
