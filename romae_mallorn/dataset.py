"""
A simple example dataset.
"""
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
#from romae.utils import gen_mask
#import joblib
import polars as pl
import numpy as np
import random
## We need to create a padded version of the full parquet file


def gen_mask(mask_ratio, pad_mask, single = False):
    """
    Function might not be super well optimized...

    Parameters
    ----------
    mask_ratio : either [min_mask, max_mask] or float
        Percentage of tokens to mask out
    pad_mask : torch.Tensor
        A boolean mask where positions in the input corresponding to
        padding have value True
    single : bool, optional

    Returns a torch.Tensor
    """
    min_ratio = np.min(mask_ratio)
    max_ratio = np.max(mask_ratio)
        
    if min_ratio < 0 or min_ratio > 1 or max_ratio < 0 or max_ratio > 1:
        raise ValueError(f"Mask ratio must be between 0 and 1, but was given "
                         f"{mask_ratio}")

    ## Modify here so that the mask_ratio is uniformly chosen between min_mask and max_mask
    ## One ratio per example in the batch
    ratio = np.random.uniform(low=min_ratio, high=max_ratio) 
    #this should have one ratio per batch but idk if it will work, size=pad_mask.shape[0])
    # this does not work
    
    per_sample_n = (~pad_mask).sum(dim=1)
    n_masked_per_sample = (per_sample_n * ratio).ceil().int()
    
    mask = torch.zeros(pad_mask.shape, dtype=torch.bool, device=pad_mask.device)
    for i in range(pad_mask.shape[0]):
        idxs = random.sample(range(per_sample_n[i].item()), n_masked_per_sample[i].item())
        for j in idxs:
            mask[i, j] = True
    if single:
        max_masked = torch.tensor(pad_mask.shape[1] * ratio).ceil().int()
    else:
        max_masked = n_masked_per_sample.max()
    diff_from_max = (n_masked_per_sample - max_masked)
    for i in range(diff_from_max.shape[0]):
        for j in range(pad_mask.shape[1] + diff_from_max[i], pad_mask.shape[1]):
            mask[i, j] = True

    return mask

def gen_mask_window(mask_ratio, pad_mask, single=False, window_ratio=0.5, window_size_range=(5, 20)):
    """
    Function that masks time-series with a mix of consecutive windows and random positions.
    
    Parameters
    ----------
    mask_ratio : either [min_mask, max_mask] or float
        Percentage of tokens to mask out
    pad_mask : torch.Tensor
        A boolean mask where positions in the input corresponding to
        padding have value True
    single : bool, optional
    window_ratio : float, optional (default=0.5)
        Proportion of masked tokens that should be in consecutive windows (vs random).
        E.g., 0.5 means half the masks are windows, half are random.
    window_size_range : tuple, optional (default=(3, 10))
        (min_size, max_size) for the length of consecutive windows
        
    Returns
    -------
    torch.Tensor
        Boolean mask indicating which positions are masked
    """
    min_ratio = np.min(mask_ratio)
    max_ratio = np.max(mask_ratio)
        
    if min_ratio < 0 or min_ratio > 1 or max_ratio < 0 or max_ratio > 1:
        raise ValueError(f"Mask ratio must be between 0 and 1, but was given "
                         f"{mask_ratio}")
    
    if window_ratio < 0 or window_ratio > 1:
        raise ValueError(f"Window ratio must be between 0 and 1, but was given "
                         f"{window_ratio}")
    
    # Sample mask ratio uniformly 
    ## 
    ratio =  np.random.uniform(low=min_ratio, high=max_ratio)
    
    per_sample_n = (~pad_mask).sum(dim=1)
    n_masked_per_sample = (per_sample_n * ratio).ceil().int()
    # print("n_masked_per_sample")
    # print(n_masked_per_sample)
    mask = torch.zeros(pad_mask.shape, dtype=torch.bool, device=pad_mask.device)
    
    for i in range(pad_mask.shape[0]):
        n_to_mask = n_masked_per_sample[i].item()
        # print("n to mask")
        # print(n_to_mask)
        
        available_positions = per_sample_n[i].item()
        
        # Split between window masks and random masks
        n_window_masks = int(n_to_mask * window_ratio)
        #n_random_masks = n_to_mask - n_window_masks
        #assert(n_window_masks+n_random_masks == n_to_mask)
        
        masked_positions = set()
        
        # Create window masks (consecutive positions)
        while (len(masked_positions) < n_window_masks) & ( (n_window_masks - len(masked_positions)) > window_size_range[0]):
            # Random window size 5- 24 -22
            #print(window_size_range[0], n_window_masks , len(masked_positions))
            
            window_size = np.random.randint(window_size_range[0], 
                                           min(window_size_range[1], n_window_masks - len(masked_positions)) + 1)
            
            # Random starting position (ensure window fits within available positions)
            if available_positions - window_size > 0:
                start_pos = np.random.randint(0, available_positions - window_size + 1)
            else:
                start_pos = 0
                window_size = min(window_size, available_positions)
            
            # Add window positions
            for pos in range(start_pos, min(start_pos + window_size, available_positions)):
                if len(masked_positions) < n_window_masks:
                    masked_positions.add(pos)

        # print("masked with windows")
        # print(len(masked_positions))
        n_random_masks = n_to_mask - len(masked_positions)
        # Create random masks
        available_for_random = set(range(available_positions)) - masked_positions
        if len(available_for_random) > 0 and n_random_masks > 0:
            random_positions = np.random.choice(
                list(available_for_random), 
                size=min(n_random_masks, len(available_for_random)), 
                replace=False
            )
            masked_positions.update(random_positions)

        # print("masked")
        # print(len(masked_positions))
        # Apply mask
        for j in masked_positions:
            mask[i, j] = True

    
    # Handle padding alignment
    if single:
        max_masked = torch.tensor(pad_mask.shape[1] * ratio).ceil().int()
    else:
        max_masked = n_masked_per_sample.max()

    max_masked = torch.tensor(pad_mask.shape[1] * ratio).ceil().int()

    ## Is it really what we want?
    ## We want each row to have the same amount of mask, "picking" in the padding 
    ## So we look at the one with the most masked (longuest light-curve in the batch
    ## or 'if single' just infer it from the size of the lc because the length should be that
    ## max_masked = max number of masked values in the batch
    ## diff_from_max = how many we are missing to match for each in the batch (in negative)
    ## The end of the padding is used to compensate by setting it at true 
    
    diff_from_max = (n_masked_per_sample - max_masked)
    for i in range(diff_from_max.shape[0]):
        for j in range(pad_mask.shape[1] + diff_from_max[i], pad_mask.shape[1]):
            mask[i, j] = True

    
    return mask

def padd_parquet(parqu_, col_names_to_pad=['FLUXCAL', 'FLUXCALERR', 'MJD', 'BAND']):
    ##  
    #parqu_ = pl.read_parquet('/scratch/gcontard/ELASTICC2/combined_train_parquets/train.parquet')

    ## Find max length
    #lents = [len(p) for p in parqu_['FLUXCAL']]
    maxlen = max(parqu_['FLUXCAL'].list.len()) #max(lents)
    padd_mask = False
    for col in col_names_to_pad:
        
        parqu_ = parqu_.with_columns((pl.col(col)).alias(col+"_pad"))
        
        parqu_ = parqu_.with_columns(
           pl.col(col+"_pad").list.concat(
              pl.lit(0).repeat_by(
                 maxlen - pl.col(col).list.len()
              )
           )
        )
        
        ## ADD A TRACK OF THE PADD MASK  
        if not(padd_mask):            
            parqu_ = parqu_.with_columns(
                pl.lit(False).repeat_by(
                            pl.col(col).list.len()).list.concat(pl.lit(True).repeat_by(maxlen - pl.col(col).list.len())).alias("PADD_MASK"))
            padd_mask = True
                
    ## I thought Array would be better once we padded everything but i must using this wrong.
    ## Also I'd like to fucking keep it f32 but?! why no?! let's see if this is a big issue down the line...

    ## I DONT KNOW IF WE SHOULD INCLUDE FLAGS AND ALL , ARE THERE ANY IN ELASTICC2 THAT SIM LSST?
    
    return parqu_
    
def map_bands(band_letters):
    band_dic = {
        'u': 0,
        'g': 1,
        'r': 2,
        'i': 3,
        'z': 4,
        'Y': 5,
        'y':5,
        "0" :-1
    }
    return [band_dic[l] for l in band_letters]

def reformat_bands(parqu_):
    ## This reformats the bands from letters to numbers
    ## Force return_dtype list of int32 instead of 64? 
    parqu_ = parqu_.with_columns(pl.col("BAND_pad").map_elements(map_bands, return_dtype=list[int]).alias("band_number"))
        
    return parqu_
    
class MallornDataset(Dataset):
    """
    This assumes some naming in the parquet taken as input -- maybe change to something better when we also adjust for DP1?
    """
    

    def __init__(self, parquet_file, 
                 mask_ratio = 0.5, gaussian_noise: bool = False):
        
        self.noise = gaussian_noise
        
        self.mask_ratio = mask_ratio

        #if isinstance(parquet_input, str):
        self.parquet = pl.read_parquet(parquet_file)
        # else:
        #     self.parquet = parquet_file
        
        
        self.parquet = padd_parquet(self.parquet)
        self.parquet = reformat_bands(self.parquet)
    ## Hopefully we don't need that?
    # def get_standardization_vals(self):
    #     import tqdm
    #     n_samples = self.file["data"].shape[0]
    #     means = torch.zeros(6)
    #     stds = torch.zeros(6)
    #     for i in tqdm.tqdm(range(n_samples), total=n_samples):
    #         data = self.file["data"][i]
    #         mask = self.file["mask"][i]
    #         for j in range(6):
    #             means[j] += data[:, j][mask[:, j] > 0.5].mean() / n_samples
    #             stds[j] += data[:, j][mask[:, j] > 0.5].std() / n_samples

    #     return means, stds


    
    

    def __len__(self):
        return len(self.parquet)

    def __enter__(self):
        return self

    # def __exit__(self, exc_type, exc_value, traceback):
    #     self.file.close()

    def __exit__(self, exc_type, exc_value, traceback):
        del self.parquet

    def __getitem__(self, idx):
        
        # FLuxCal here should be the DIFF flux !!
        data = torch.tensor(self.parquet["FLUXCAL_pad"][idx].to_numpy()).flatten()
        pad_mask = ~(torch.tensor(self.parquet["PADD_MASK"][idx].to_numpy())).flatten()
        # Adjust if we have alert and flags?
        # alert_mask = (torch.tensor(self.file["mask_alert"][idx]) > 0.5).flatten()
        # pad_mask[alert_mask] = True
        times = torch.tensor(self.parquet["MJD_pad"][idx].to_numpy().flatten())
        times[pad_mask] = times[pad_mask] - torch.min(times[pad_mask]) #To avoid big numbers in times
        
        #label =  self.parquet["ELASTICC_class"][idx] # THIS IS A STRING 
        bands = torch.tensor(self.parquet["band_number"][idx].to_numpy()) # this is padded
        positions = torch.stack([bands, times])
        data_var = torch.tensor(self.parquet["FLUXCALERR_pad"][idx]).flatten()
        data = torch.stack([data, data_var])
        n_nonpad = pad_mask.sum()
        positions = nn.functional.pad(positions[:, pad_mask], (0, positions.shape[1]-n_nonpad)).float()
        data = nn.functional.pad(data[:, pad_mask], (0, data.shape[1]-n_nonpad))[..., None, None].float().swapaxes(0, 1)
        pad_mask[:] = False
        pad_mask[n_nonpad:] = True
        mask = gen_mask_window(self.mask_ratio, pad_mask[None, ...], single=True).squeeze()
        if self.noise:
            data = data + torch.randn_like(data) * 0.02
        sample = {
            "values": data,
            "positions": positions,
            #"label": label,
            "mask": mask,
            "pad_mask": pad_mask
        }
        return sample


    # def get_item_label(self, idx):

    #     # TODO: Need to adjust as right now the labels are string, this might cause a mess later down the line
        
    #     # FLuxCal here should be the DIFF flux !!
    #     data = torch.tensor(self.parquet["FLUXCAL_pad"][idx].to_numpy()).flatten()
    #     pad_mask = ~(torch.tensor(self.parquet["PADD_MASK"][idx].to_numpy())).flatten()
    #     # Adjust if we have alert and flags?
    #     # alert_mask = (torch.tensor(self.file["mask_alert"][idx]) > 0.5).flatten()
    #     # pad_mask[alert_mask] = True
    #     times = torch.tensor(self.parquet["MJD_pad"][idx].to_numpy().flatten())
    #     times[pad_mask] = times[pad_mask] - torch.min(times[pad_mask]) #To avoid big numbers in times
        
    #     label =  self.parquet["ELASTICC_class"][idx] # THIS IS A STRING 
    #     bands = torch.tensor(self.parquet["band_number"][idx].to_numpy()) # this is padded
    #     positions = torch.stack([bands, times])
    #     data_var = torch.tensor(self.parquet["FLUXCALERR_pad"][idx]).flatten()
    #     data = torch.stack([data, data_var])
    #     n_nonpad = pad_mask.sum()
    #     positions = nn.functional.pad(positions[:, pad_mask], (0, positions.shape[1]-n_nonpad)).float()
    #     data = nn.functional.pad(data[:, pad_mask], (0, data.shape[1]-n_nonpad))[..., None, None].float().swapaxes(0, 1)
    #     pad_mask[:] = False
    #     pad_mask[n_nonpad:] = True
    #     mask = gen_mask(self.mask_ratio, pad_mask[None, ...], single=True).squeeze()
    #     if self.noise:
    #         data = data + torch.randn_like(data) * 0.02
    #     sample = {
    #         "values": data,
    #         "positions": positions,
    #         "label": label,
    #         "mask": mask,
    #         "pad_mask": pad_mask
    #     }
    #     return sample

  
class MallornDatasetwLabel(Dataset):
    """
    This assumes some naming in the parquet taken as input -- maybe change to something better when we also adjust for DP1?
    """
    

    def __init__(self, parquet_file, 
                 mask_ratio = 0.5, gaussian_noise: bool = False):
        
        self.noise = gaussian_noise
        
        self.mask_ratio = mask_ratio

        #if isinstance(parquet_input, str):
        self.parquet = pl.read_parquet(parquet_file)
        # else:
        #     self.parquet = parquet_file
        
        
        self.parquet = padd_parquet(self.parquet)
        self.parquet = reformat_bands(self.parquet)
    ## Hopefully we don't need that?
    # def get_standardization_vals(self):
    #     import tqdm
    #     n_samples = self.file["data"].shape[0]
    #     means = torch.zeros(6)
    #     stds = torch.zeros(6)
    #     for i in tqdm.tqdm(range(n_samples), total=n_samples):
    #         data = self.file["data"][i]
    #         mask = self.file["mask"][i]
    #         for j in range(6):
    #             means[j] += data[:, j][mask[:, j] > 0.5].mean() / n_samples
    #             stds[j] += data[:, j][mask[:, j] > 0.5].std() / n_samples

    #     return means, stds


    
    

    def __len__(self):
        return len(self.parquet)

    def __enter__(self):
        return self

    # def __exit__(self, exc_type, exc_value, traceback):
    #     self.file.close()

    def __exit__(self, exc_type, exc_value, traceback):
        del self.parquet

    def __getitem__(self, idx):
        
        # FLuxCal here should be the DIFF flux !!
        data = torch.tensor(self.parquet["FLUXCAL_pad"][idx].to_numpy()).flatten()
        pad_mask = ~(torch.tensor(self.parquet["PADD_MASK"][idx].to_numpy())).flatten()
        # Adjust if we have alert and flags?
        # alert_mask = (torch.tensor(self.file["mask_alert"][idx]) > 0.5).flatten()
        # pad_mask[alert_mask] = True
        times = torch.tensor(self.parquet["MJD_pad"][idx].to_numpy().flatten())
        times[pad_mask] = times[pad_mask] - torch.min(times[pad_mask]) #To avoid big numbers in times
        
        label =  self.parquet["binary_class"][idx] # 
        bands = torch.tensor(self.parquet["band_number"][idx].to_numpy()) # this is padded
        positions = torch.stack([bands, times])
        data_var = torch.tensor(self.parquet["FLUXCALERR_pad"][idx]).flatten()
        data = torch.stack([data, data_var])
        n_nonpad = pad_mask.sum()
        positions = nn.functional.pad(positions[:, pad_mask], (0, positions.shape[1]-n_nonpad)).float()
        data = nn.functional.pad(data[:, pad_mask], (0, data.shape[1]-n_nonpad))[..., None, None].float().swapaxes(0, 1)
        pad_mask[:] = False
        pad_mask[n_nonpad:] = True
        mask = gen_mask_window(self.mask_ratio, pad_mask[None, ...], single=True).squeeze()
        if self.noise:
            data = data + torch.randn_like(data) * 0.02
        sample = {
            "values": data,
            "positions": positions,
            "label": label,
            "mask": mask,
            "pad_mask": pad_mask
        }
        return sample


