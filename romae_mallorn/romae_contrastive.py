
import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from romae.model import RoMAEForPreTraining, RoMAEForPreTrainingConfig, EncoderConfig, RoMAEBase, Encoder

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

"""
RoMAE Contrastive Learning

Extends RoMAEForPreTraining to add contrastive learning capabilities.
"""



# Allow utilization of tensor cores
torch.backends.cuda.matmul.allow_tf32 = True

class ProjectionHead(nn.Module):
    """Projection head for contrastive learning"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        ## Check
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

        
        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        
        self.mlp2 = nn.Sequential(
              nn.BatchNorm1d(hidden_dim),
              nn.ReLU(inplace=True),
              nn.Linear(hidden_dim, output_dim)
            )
    
    def forward(self, x):
        outml1 = self.mlp1(x)
        return self.mlp2(outml1)

    def get_emb(self,x):
        return self.mlp1(x)



class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None, n_views: int = 2):
        batch_size = features.shape[0] // n_views # n_views : number of new-x generated from noising function of choice
        features = F.normalize(features, dim=1)
        #print("Feature shape")
        #print(features.shape)
        similarity_matrix = torch.matmul(features, features.T) # eq to cosine_sim because of normalization
        
        #cos_sim = F.cosine_similarity(features[:,None,:], features[None,:,:], dim=-1)
        #print("Similarity matrix shape")
        #print(similarity_matrix.shape)
        
        if labels is None:
            labels_matrix = torch.cat([torch.arange(batch_size) for _ in range(n_views)], dim=0)
            labels_matrix = (labels_matrix.unsqueeze(0) == labels_matrix.unsqueeze(1)).float().to(features.device)
            #print("When label is None")
            #print(labels_matrix.shape) batchsize*2 x batchsize*2 
        else:
            labels_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(features.device)
            #print(labels_matrix.shape)
            ## I"m not sure this will actually work properly -- it might be too driven by the exact same examples?
            ## Also need to ignore/mask the unsupervised ones because we dont want to distinguish them, altho it might be funny if we do
            ## as in it's a test to run but make the problem/loss more confusing for the network esp since it's a large frac of the dataset?

        mask = torch.eye(labels_matrix.shape[0], dtype=torch.bool, device=features.device)
        labels_matrix = labels_matrix[~mask].view(labels_matrix.shape[0], -1)
        #print("Not diagonal")
        #print(labels_matrix.shape)

        ## Compute similarity off diagonal 
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        #print(similarity_matrix.shape)
        positives = similarity_matrix[labels_matrix.bool()].view(labels_matrix.shape[0], -1)
        #print("Positive shape")
        #print(positives.shape)
        negatives = similarity_matrix[~labels_matrix.bool()].view(similarity_matrix.shape[0], -1)
        #print("Negative shape")
        #print(negatives.shape)

        ## Cross Entropy Loss: for each row we want the 1st elem (col) to have the highest similarity since it's th ecounter part
        ## hence label==0
        logits = torch.cat([positives, negatives], dim=1) / self.temperature
        target_labels = torch.zeros(logits.shape[0], dtype=torch.long, device=features.device)
        
        return self.criterion(logits, target_labels)



def scramble_according_labels(view1, labels):
    ## view1 are features of examples with labels labels
    ## This returns a shuffle of view1 such that labels are preserved.

    ## I'm not making sure that all i in newview are different to view1
    
    newview = torch.zeros(view1.shape).to(view1.device)
    for k in torch.unique(labels):
        idx_where_k_in_batch = torch.where(labels==k)[0].cpu().numpy()
        #pick_idx = random.choices(list(idx_where_k_in_batch), k=len(idx_where_k_in_batch)) # not sure about the replacement etc etc
        pick_idx = np.random.choice(list(idx_where_k_in_batch), size=len(idx_where_k_in_batch), replace=True)
        #print(pick_idx)
        newview[idx_where_k_in_batch,:] = view1[pick_idx,:]
        
    return newview
    
class RoMAEPreTrainingContrastive(RoMAEForPreTraining):
    """
    Extends RoMAEForPreTraining to add contrastive learning.
    
    This subclass overrides the forward method to return a dictionary
    containing encoded representations, projections, and losses.
    """
    
    def __init__(
        self,
        config: RoMAEForPreTrainingConfig,
        contrastive_config: MallornConfigContrastive,
        augmentation_fn=None
    ):
        """
        Args:
            config: RoMAEForPreTrainingConfig
            contrastive_config: ContrastiveConfig
            augmentation_fn: Function to augment inputs
        """
        super().__init__(config)
        
        self.contrastive_config = contrastive_config
        
        # Determine projection input dimension
        d_model = self.encoder.config.d_model
        projection_input_dim = contrastive_config.cls_contrastive_dim if contrastive_config.cls_contrastive_dim else d_model
        
        # Projection head for contrastive learning
        self.projection_head = ProjectionHead(
            input_dim=projection_input_dim,
            hidden_dim=contrastive_config.projection_hidden_dim,
            output_dim=contrastive_config.projection_dim
        )
        
        # Contrastive losses
        self.aug_contrast_loss = ContrastiveLoss(temperature=contrastive_config.temperature)
        self.class_contrast_loss = ContrastiveLoss(temperature=contrastive_config.temperature)
        
        # Augmentation function
        self.augmentation_fn = augmentation_fn or (lambda x, y: x)

        self.compute_contrastive = True # eeeeeehh...
        self.decode= contrastive_config.decode


    def forward_cls(self,
                    values: torch.Tensor,
                    mask: torch.Tensor,
                    positions: torch.Tensor,
                    pad_mask: Optional[torch.Tensor] = None,
                    decode: Optional[bool] = True):
        '''
           Returns a dictionnary with the embeddings (including CLS), reconstructed masked, and loss
        '''
        b = values.shape[0]
        npd = self.config.n_pos_dims
        # Convert input to a sequence of tubelets
        x = patchify(self.config.tubelet_size, values)

        # Extract all the values that are being masked out
        m_x = x[mask].reshape(b, -1, x.shape[-1])
        m_positions = positions[mask[:, None, ].expand(-1, npd, -1)].reshape(b, npd, -1)
        m_pad_mask = None
        if pad_mask is not None:
            m_pad_mask = pad_mask[mask].reshape(b, -1)

        # Now get all the values that are not masked out
        x = x[~mask].reshape(b, -1, x.shape[-1])
        positions = positions[~mask[:, None, ...].expand(-1, npd, -1)].reshape(b, npd, -1)
        if pad_mask is not None:
            pad_mask = pad_mask[~mask].reshape(b, -1)

        # Project into embeddings
        x = self.projection(x)
        # Add classification token to the beginning of all relevant tensors
        x, positions, pad_mask = self.add_cls(x, positions, pad_mask)

        x = self.inpt_pos_dropout(self.encoder_inpt_pos_embedding(x, positions, ~mask))

        attn_mask = _get_attn_mask(x.shape, x.device, pad_mask)

        # Encoder forward pass
        embedd = self.encoder(
            x,
            positions=positions,
            pos_encoding=self.encoder_attn_pos_embedding,
            attn_mask=attn_mask
        )
        # Project tokens from the encoder dimension to decoder dimension
        embedd_proj = self.encoder_decoder_proj(embedd)

        mask_tokens = self.mask_token.expand(b, m_x.shape[1], -1)

        # Apply input positional encodings to our MASK tokens.
        mask_tokens = self.inpt_pos_dropout(
            self.decoder_inpt_pos_embedding(mask_tokens, m_positions, mask)
        )

        # Append MASK token and positional information
        all_ = torch.cat([embedd_proj, mask_tokens], dim=1)
        positions = torch.cat([positions, m_positions], dim=2)
        if pad_mask is not None:
            pad_mask = torch.cat([pad_mask, m_pad_mask], dim=1)

        outputs = {}
        outputs["embeddings"] = embedd
        outputs["logits"] = None
        outputs["recon_loss"] = None
        
        if decode:
                # Get our new attention and padding masks
            attn_mask = _get_attn_mask(all_.shape, all_.device, pad_mask)

            # Decoder forward pass
            x = self.decoder(
                all_,
                positions=positions,
                pos_encoding=self.decoder_attn_pos_embedding,
                attn_mask=attn_mask
            )
            m_x = self.normalize_targets(m_x)
            x = x[:, -m_x.shape[-2]:]

            logits, loss = None, None
            if m_x.shape[1] != 0:
                logits, loss = self.apply_head_loss(x, m_x)
                if m_pad_mask is not None:
                    # Remove loss from padding:
                    loss[m_pad_mask] = 0
                    loss = loss.mean()
            
                outputs["logits"]=logits
                outputs["recon_loss"]=loss

        

        # We reset the positional embedding caches to avoid
        # inter-loop dependencies in the Trainer, which break torch compile.
        self.reset_pos_cache()
        
        ## all_ : cls, embeddings + mask tokens, as 'decoder input' (so potentially a diffrent size than encoding)
        ## Embedd: cls + embedd  in encoding
        ## logits: predictions for masked_values
    
        return outputs

    
    def extract_cls_and_project(
        self,
        cls_full: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract and project CLS token.
        
        Returns:
            cls_contrastive: CLS portion for contrastive
            projections: Projected features
            cls_full: Full CLS token
        """
        # 
        
        # Split CLS if configured
        if self.contrastive_config.cls_contrastive_dim:
            cls_contrastive = cls_full[:, :self.contrastive_config.cls_contrastive_dim]
        else:
            cls_contrastive = cls_full
        
        # Project for contrastive learning
        projections = self.projection_head(cls_contrastive)
        
        return cls_contrastive, projections, cls_full
    
    def forward_fullinfo(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        compute_contrastive: bool = True,
        decode: bool = True
    ):
        """
        Forward pass with contrastive learning.
        
        Args:
            values: Input values (batch, time, channel, height, width)
            mask: Masked from MaskedDataset for MAE
            positions: Positional encodings
            pad_mask: Optional padding mask
            labels: Class labels for supervised contrastive loss
            compute_contrastive: If False, acts like normal RoMAEForPreTraining
        
        Returns:
            Dictionary with:
                - logits: Reconstruction logits (from parent)
                - loss: Combined loss (reconstruction + contrastive)
                - encoded: Encoded representations
                - projections: Contrastive projections
                - loss_dict: Breakdown of losses
        """
        if not compute_contrastive:
            # Act like normal RoMAEForPreTraining
            #logits, recon_loss = super().forward(values, positions, pad_mask)
            return self.forward_cls(values, positions, pad_mask, decode=True)
            
        
        # Generate augmented views

        ## GABY: Add something to have multiple augmentation_fn? Or just do masking for now
        
        values_aug1, newmask1, _, _ = self.augmentation_fn(values, mask, positions, pad_mask)# labels)
        values_aug2, newmask2, _, _ = self.augmentation_fn(values,  mask, positions, pad_mask)
        
        # Extract CLS and project for all views
        #_, proj_orig, _ = self.extract_cls_and_project(values, positions, pad_mask)
        # GABY TODO: Maybe here we should keep the CLS/classic forward if we train with the regular RoMAE on top??
        outputs_og = self.forward_cls(values, mask, positions, pad_mask, decode= (self.contrastive_config.recon_weight > 0))
        cls_full = outputs_og['embeddings'][:, 0, :]
        _, proj_orig, _ = self.extract_cls_and_project(cls_full)

        ## We could decide to remove the mask form the DataLoader; or to work on top of it?
        ## GABY: TODO  THINK ABOUT THIS --- This makes aug1/aug2 a different "size" as OG but maybe this is good?
        outputs_aug1 = self.forward_cls(values_aug1, newmask1, positions, pad_mask, decode=False)
        _, proj_aug1, _ = self.extract_cls_and_project(outputs_aug1['embeddings'][:,0,:])
        del outputs_aug1
        
        outputs_aug2 = self.forward_cls(values_aug2, newmask2, positions, pad_mask, decode=False)
        _, proj_aug2, _ = self.extract_cls_and_project(outputs_aug2['embeddings'][:,0,:])
        del outputs_aug2
        
        # Concatenate augmented projections
        proj_concat = torch.cat([proj_aug1, proj_aug2], dim=0)
        
        loss_dict = {}
        total_loss = 0.0
        
        # 1. Augmentation-based contrastive loss
        if self.contrastive_config.aug_contrast_weight > 0:
            aug_loss = self.aug_contrast_loss(proj_concat, None, self.contrastive_config.n_views)
            loss_dict['aug_contrast_loss'] = aug_loss.item()
            total_loss += self.contrastive_config.aug_contrast_weight * aug_loss
        
        # 2. Class-based contrastive loss
        if self.contrastive_config.class_contrast_weight > 0 and labels is not None:

            ## GABY URGENT: HANDLE THE UNSUPERVISED ONES !!!! WE DONT WANT TO CONTRASTIVE ON THEM?
            ## AND ALSO MAKE SURE THAT THERE IS A POSITIVE COUNTERPART OR ACCESS TO THE FULL DATASET
            ## IN MALLORNDATASET

            ## DOUBLE CHECK THIS IS CORRECT???
            proj_aug2_scrambled = scramble_according_labels(proj_aug2, labels)
            ## Lets remove the unsupervised
            proj_aug2_scrambled = proj_aug2_scrambled[labels!=-1]
            proj_aug1_sup = proj_aug1[labels!=-1]
            proj_concat_class = torch.cat([proj_aug1_sup, proj_aug2_scrambled], dim=0)

            ## DOUBLE CHECK THIS IS CORRECT???
            
            #labels_concat = torch.cat([labels, labels], dim=0)
            class_loss = self.class_contrast_loss(proj_concat_class, None, self.contrastive_config.n_views)
            # labels_concat = torch.cat([labels, labels], dim=0)
            # class_loss = self.class_contrast_loss(proj_concat, labels_concat, self.contrastive_config.n_views)
            loss_dict['class_contrast_loss'] = class_loss.item()
            total_loss += self.contrastive_config.class_contrast_weight * class_loss
        
        # 3. Reconstruction loss (from parent)
        if self.contrastive_config.recon_weight > 0:
            ## Could be less expensive because this is an extra fwd pass + GPU mem then
            # logits, recon_loss = super().forward(values, positions, pad_mask)
            loss_dict['recon_loss'] = outputs_og['recon_loss'].item()
            total_loss += self.contrastive_config.recon_weight * outputs_og['recon_loss'].item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return {
            'logits': outputs_og['logits'],
            'loss': total_loss,
            'encoded': outputs_og['embeddings'],
            'projections': proj_orig,
            'aug1_projections': proj_aug1,
            'aug2_projections': proj_aug2,
            'loss_dict': loss_dict
        }

    
    def forward(self, values: torch.Tensor, mask: torch.Tensor,
                positions: torch.Tensor, pad_mask=None,
                label=None, *_, **__):
        full_ = self.forward_fullinfo(values, mask, positions, pad_mask, labels, self.compute_contrastive, self.decode)
        return full_['logits'], full['loss']
        



# ==================== Augmentation Functions ====================

#

class RandomMasking:
    """Random masking augmentation"""
    def __init__(self, mask_ratio: float = 0.5):
        self.mask_ratio = mask_ratio
    def __call__(self, values, mask, positions, pad_mask):
        # extra_mask = torch.rand_like(mask, dtype=float) > self.mask_ratio
        # new_mask = torch.clone(mask)
        # new_mask[extra_mask] = True

        ## Need to preserve rectangularity so we need to look at pad_mask to ensure things keep kosher
        ## Just use the gen_mask function defined in dataset.py for Mallorn; maybe we can increase the mask_ratio?
        new_mask = gen_mask(self.mask_ratio, pad_mask, single=True).squeeze()
        
        ## No need to replace values and positions because the masking should be handled by RoMAE...?
        ## And keep it consistant with an augmentation function that adds noise or somthing
        return values, new_mask, positions, pad_mask
        #return x * (torch.rand_like(x) > self.mask_ratio).float()

### WINDOWMASKING?