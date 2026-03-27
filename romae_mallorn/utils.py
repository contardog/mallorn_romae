from romae.utils import get_encoder_size

from romae_mallorn.romae_contrastive import RoMAEPreTrainingContrastive, RandomMasking
from romae_mallorn.dataset import MallornDatasetwLabelTrimMask
from romae_mallorn.utils import override_encoder_size
from romae_mallorn.env_config import MallornConfigContrastiveEnv



from romae_mallorn.samplers import PositiveGuaranteedSampler

import json

from pathlib import Path

def load_from_checkpoint_contrastive(checkpoint_dir, model_cls, model_config, contrastive_config):
    """
    Load a model from a checkpoint.

    Parameters
    ----------
    checkpoint_dir : str
    model_cls
        The actual uninitialized class of the model being loaded
    model_config
        Uninitialized configuration class of the model being loaded

    Returns
    -------
    model
        The provided model_cls loaded with the weights and
        configuration present in the checkpoint directory.
    """
    checkpoint_dir = Path(checkpoint_dir)
    # Load model configuration
    with open(checkpoint_dir/"model_config.json", "r") as f:
        config_json = json.load(f)
    print(config_json)
    config = model_config(**config_json)

    # Initialize the model class using the loaded configuration
    model = model_cls(config, contrastive_config)
    # Load the model weights from the checkpoint
    model.load_weights(checkpoint_dir)
    return model


def override_encoder_size(size: str):
    """Get the parameters of a specific RoMAE model encoder size.
    """
    try:
        ## Check RoMAE's code for the already predefined sizes with larger dimension of embeddings 
        return get_encoder_size(size)
    except:
        
        match size:
            case "tiny-midshallow10":
                return {
                    "d_model": 180,
                    "nhead": 3,
                    "depth": 10 #Use to be 2
                }
            case "verytiny-10":
                return {
                    "d_model": 60,
                    "nhead": 3,
                    "depth": 10 #Use to be 2
                }
            case "verytiny-8":
                return {
                    "d_model": 60,
                    "nhead": 3,
                    "depth": 8 #Use to be 2
                }
            case "verytiny-6":
                return {
                    "d_model": 60,
                    "nhead": 3,
                    "depth": 6 #Use to be 2
                }
            case "tiny-midshallow8":
                return {
                    "d_model": 180,
                    "nhead": 3,
                    "depth": 10 #Use to be 2
                }
            case "tiny-midshallow_real8":
                return {
                    "d_model": 180,
                    "nhead": 3,
                    "depth": 8 #Use to be 2
                }
            case "tiny-midshallow_6":
                return {
                    "d_model": 180,
                    "nhead": 3,
                    "depth": 6 #Use to be 2
                }
            case "tinyer-midshallow_real8":
                return {
                    "d_model": 120,
                    "nhead": 3,
                    "depth": 8 #Use to be 2
                }
            case "tinyer-midshallow_6":
                return {
                    "d_model": 120,
                    "nhead": 3,
                    "depth": 6 #Use to be 2
                }
            case "tinyer-midshallow":
                return {
                    "d_model": 120,
                    "nhead": 3,
                    "depth": 10 #Use to be 2
                }
            case "very-tiny-shallow":
                return {
                    "d_model": 60,
                    "nhead": 3,
                    "depth": 6 #Use to be 2
                }
            case "very-tiny":
                return {
                    "d_model": 60,
                    "nhead": 3,
                    "depth": 12
                }
            case "super-tiny":
                return {
                    "d_model": 20,
                    "nhead": 3,
                    "depth": 12
                }
            case "super-tiny-shallow":
                return {
                    "d_model": 20,
                    "nhead": 3,
                    "depth": 2
                }
            
            case "brutally-mini":
                return {
                    "d_model": 8,
                    "nhead": 3,
                    "depth": 12
                }
            
            # case "brutally-mini-shallow":
            #     return {
            #         "d_model": 8,
            #         "nhead": 3,
            #         "depth": 2
            #     }
            case "2d": # So, this *runs* but it is probably not a good idea. Could we get smaller embeddings as we go tho?
                return {
                    "d_model": 2,
                    "nhead": 3,
                    "depth": 12
                }
            # case "2d-shallow":
            #     return {
            #         "d_model": 2,
            #         "nhead": 3,
            #         "depth": 2
            #     }
            case _:
                raise ValueError(f"Unknown encoder size: {size}")