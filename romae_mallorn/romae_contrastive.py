
import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from romae.model import RoMAEForPreTraining, RoMAEForPreTrainingConfig, EncoderConfig, RoMAEBase, Encoder

from romae.model import _get_attn_mask
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

import wandb

"""
RoMAE Contrastive Learning

Extends RoMAEForPreTraining to add contrastive learning capabilities.
"""



# Allow utilization of tensor cores
torch.backends.cuda.matmul.allow_tf32 = True

class OGProjectionHead(nn.Module):
    """Projection head for contrastive learning"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        
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


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        #self.mlp1 = nn.Linear(input_dim, hidden_dim)
        
        self.mlp2 = nn.Sequential(
              nn.BatchNorm1d(input_dim),
              nn.ReLU(inplace=True),
              nn.Linear(input_dim, output_dim)
            )
    
    def forward(self, x):        
        return self.mlp2(x)

    def get_emb(self,x):
        print("Warning: This ProjectionHead only has one layer, so calling this function does not make a lot of sense; this returns the input for consistency with other ProjectionHead and simCLR suggestion of not using the final embedding")
        return x


class OldContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None, n_views: int = 2):
    
        
        batch_size = features.shape[0] // n_views # n_views : number of new-x generated from noising function of choice
            
        if (batch_size>0):
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

            if labels_matrix.shape[0]>0:
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
            else:
                print('this should not happen')
                return torch.Tensor([0])
            
        else:
            print('this should not happen either?')
            return torch.Tensor([0])

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07,
                 scale_by_temperature: bool = True,
                 unsup_in_denominator: bool = False,):
        super().__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature
        self.unsup_in_denominator = unsup_in_denominator


    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None, n_views: int = 2):
        """
        features : (N, D) — already the contrastive-head output (e.g. 32-dim slice)
        labels   : (N,)   — 1 (positive), 0 (negative), -1 (unsupervised)
                            if None, falls back to SimCLR-style self-supervised loss using n_views
        n_views  : only used when labels is None
        """
        device = features.device
        features = F.normalize(features, dim=1)
        N = features.shape[0]

        # ------------------------------------------------------------------ #
        # Build the mask of valid (anchor, positive) pairs                   #
        # ------------------------------------------------------------------ #
        if labels is None:
            # SimCLR fallback: views of the same sample are positives
            batch_size = N // n_views
            base_ids = torch.arange(batch_size, device=device)
            ids = torch.cat([base_ids for _ in range(n_views)])          # (N,)
            pos_mask = (ids.unsqueeze(0) == ids.unsqueeze(1)).float()    # (N, N)

        else:
            # SupCon: same label => positive, BUT:
            #   - unsupervised (-1) anchors are skipped entirely
            #   - unsupervised (-1) examples are also excluded as positives
            #     (we don't know their class, so we can't call them positive
            #      for anything — they still appear in the denominator as negatives)

            ## This could be questionable? Let'see in practice, we could also drop the unsupervised from the denominator entirely
                        
                 
            pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # (N, N)
            # zero out any pair involving an unsupervised sample
            sup_mask = (labels != -1).float()                                # (N,)
            pos_mask = pos_mask * sup_mask.unsqueeze(0) * sup_mask.unsqueeze(1)

        # Remove self-similarity from positives (diagonal)
        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        pos_mask[self_mask] = 0.0

        """From authors code: 
            The contrastive loss then takes the following form:
        
            L = \sum_{i} L_i
        
          where each L_i is computed as:
        
            L_i = -\tau * \sum_{k \in P(i)} \log(p_{ik})    (1)
            ##Note : the code below also has this extra tau (temperature)
            this was implemented if i understand correctly to control/compensate the 
            gradient scaling because of the 1/tau backed in the loss;
            so now changing the temperature does not require to modify the learning rate to properly investigate the role of tau
            
        
          where P(i) is the set of positives for entry i (distinct from i) and where:
        
                               \exp(f_i^T f_k / \tau)
            p_{ik} = ----------------------------------------                        (2)
                     \sum_{j \in A(i)} \exp(f_i^T f_j / \tau)
        
          where A(i) is the set of all positives or negatives (distinct from i). `i` is
          the anchor, and \tau is the temperature.

          NOTE FROM THEIR CODE: Actually different ways to compute denominator (all, only negative, one positive, 
          see their code in tensorflow); also some consideration in order of operations...

          We'll go for All.
        
          This maximizes the likelihood of a given (anchor, positive) pair with
          respect to all possible pairs where the first member is the anchor and the
          second member is a positive or a negative. """
        
        # ------------------------------------------------------------------ #
        # Compute logits                                                      #
        # ------------------------------------------------------------------ #
        sim = torch.matmul(features, features.T) / self.temperature          # (N, N)
        # For numerical stability, subtract row max (like logsumexp trick)
        sim = sim - sim.detach().max(dim=1, keepdim=True).values

        # Mask out self-similarity from the denominator too
        exp_sim = torch.exp(sim)
        #exp_sim_no_self = exp_sim * (~self_mask).float()                     # (N, N)
        # ------------------------------------------------------------------ #
        # Build denominator mask                                              #
        # ------------------------------------------------------------------ #
        # Always exclude self from denominator
        denom_mask = (~self_mask).float()                                     # (N, N)

        if labels is not None and not self.unsup_in_denominator:
            # Restrict denominator to labeled examples only (columns only —
            # rows are handled by the has_positive anchor gate below)
            sup_col_mask = (labels != -1).float().unsqueeze(0).expand(N, -1) # (N, N)
            denom_mask = denom_mask * sup_col_mask

        exp_sim_denom = exp_sim * denom_mask

        #

        # ------------------------------------------------------------------ #
        # SupCon loss                                                         #
        # ------------------------------------------------------------------ #
        # For each anchor i, sum log-softmax over its positives p:
        #   L_i = -1/|P(i)| * sum_{p in P(i)} [ sim(i,p) - log(sum_{a != i} exp(sim(i,a))) ]
        #
        # log_prob[i, j] = sim(i,j) - log(sum_{a != i} exp(sim(i,a)))
        
        log_denom = torch.log(exp_sim_denom.sum(dim=1, keepdim=True) + 1e-9)  # (N, 1)
        log_prob = sim - log_denom                                               # (N, N)

        # Mean log-prob over positives for each anchor
        n_positives = pos_mask.sum(dim=1)                                    # (N,)

        # Only compute loss for anchors that actually have at least one positive
        has_positive = n_positives > 0

        if labels is not None:
            # Also skip unsupervised anchors
            has_positive = has_positive & (labels != -1)

        if not has_positive.any():
            # Degenerate batch (shouldn't happen with your K>=5 guarantee)
            return features.sum() * 0.0   # zero loss, keeps grad graph alive

        loss_per_anchor = -(pos_mask * log_prob).sum(dim=1) / (n_positives + 1e-9)  # (N,)
        loss = loss_per_anchor[has_positive].mean()
        if self.scale_by_temperature:
            loss = loss * self.temperature
            
        return loss


def scramble_according_labels(view1, labels):
    ## view1 are features of examples with labels labels
    ## This returns a shuffle of view1 such that labels are preserved.

    ## I'm not making sure that all i in newview are different to view1
    # 
    # lst = list(range(5))
    # shfld = lst[:]
    # while any(lst[i] == shfld[i] for i in range(len(lst))):
    #     random.shuffle(shfld)
        
    newview = torch.zeros(view1.shape).to(view1.device)
    print('In scramble:')
    print(torch.unique(labels, return_counts=True))
    for k in torch.unique(labels):
        idx_where_k_in_batch = torch.where(labels==k)[0].cpu().numpy()
        #pick_idx = random.choices(list(idx_where_k_in_batch), k=len(idx_where_k_in_batch)) # not sure about the replacement etc etc
        #pick_idx = np.random.choice(list(idx_where_k_in_batch), size=len(idx_where_k_in_batch), replace=True)
        #print(pick_idx)
        lst = list(idx_where_k_in_batch)
        pick_idx = lst[:]
        if len(idx_where_k_in_batch)>1:
            while any(lst[i]==pick_idx[i] for i in range(len(lst))): # is this faster on numpy array operation? boh
                random.shuffle(pick_idx)
        newview[idx_where_k_in_batch,:] = view1[pick_idx,:]
        
    return newview
    
class RoMAEPreTrainingContrastiveInfoNCE(RoMAEForPreTraining):
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
        self.aug_contrast_loss = OldContrastiveLoss(temperature=contrastive_config.temperature)
        self.class_contrast_loss = OldContrastiveLoss(temperature=contrastive_config.temperature)
        
        # Augmentation function
        self.augmentation_fn = augmentation_fn or (lambda x, y: x)

        self.compute_contrastive = True # eeeeeehh...
        self.decode= contrastive_config.decode

        self.batch_counter=0


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

        print(x.shape)
        print(mask.shape)
        
        if len(mask.shape)==1:
            mask = mask.unsqueeze(0)
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
        #check if labels are not just unsup
        nolabelsup = False
        if labels is not None:
            # print('Labels')
            # print(torch.unique(labels))

            if labels[labels==-1].shape[0]==0:
                nolabelsup = True
                # print('no label sup')
            
        if (not compute_contrastive): # | nolabelsup:
            # Act like normal RoMAEForPreTraining
            """ This needs fixing because it's not returning the same dict as the other branch """
            #print("ypassing contrastive loss -- this might break some stuff because it's not returning a dictionnary actually")
            
            return self.forward_cls(values, mask, positions, pad_mask, decode=True)

        # print('Forward')
        # print(values.shape)

        # Generate augmented views

        ## GABY: Add something to have multiple augmentation_fn? Or just do masking for now
        
        values_aug1, newmask1, _, _ = self.augmentation_fn(values, mask, positions, pad_mask)# labels)
        values_aug2, newmask2, _, _ = self.augmentation_fn(values,  mask, positions, pad_mask)
        
        # Extract CLS and project for all views
        #_, proj_orig, _ = self.extract_cls_and_project(values, positions, pad_mask)
        # GABY TODO: Maybe here we should keep the CLS/classic forward if we train with the regular RoMAE on top??

        
        ## We could decide to remove the mask form the DataLoader; or to work on top of it?
        ## GABY: TODO  THINK ABOUT THIS --- This makes aug1/aug2 a different "size" as OG but maybe this is good?
        
        outputs_aug1 = self.forward_cls(values_aug1, newmask1, positions, pad_mask, decode=(self.contrastive_config.recon_weight > 0))
        _, proj_aug1, _ = self.extract_cls_and_project(outputs_aug1['embeddings'][:,0,:])
        #del outputs_aug1
        
        outputs_aug2 = self.forward_cls(values_aug2, newmask2, positions, pad_mask, decode=False)
        _, proj_aug2, _ = self.extract_cls_and_project(outputs_aug2['embeddings'][:,0,:])
        del outputs_aug2
        
        # Concatenate augmented projections
        proj_concat = torch.cat([proj_aug1, proj_aug2], dim=0)
        
        loss_dict = {}
        total_loss = 0.0 #torch.Tensor([0.0]).to(values.device)
        
        # 1. Augmentation-based contrastive loss
        if self.contrastive_config.aug_contrast_weight > 0:
            if proj_concat.shape[0]>0: # Not sure why this wouldnt be the case? // maybe remove
                # print('in augment contrast')
                aug_loss = self.aug_contrast_loss(proj_concat, None, self.contrastive_config.n_views)
                loss_dict['aug_contrast_loss'] = aug_loss.item()
                total_loss += self.contrastive_config.aug_contrast_weight * aug_loss        
            # else:
            #     loss_dict['aug_contrast_loss'] = 0
                #total_loss += 0
            
        # 2. Class-based contrastive loss
        if self.contrastive_config.class_contrast_weight > 0 and labels is not None:


            if proj_aug2.shape[0]>0: #  Not sure why this wouldnt be the case? // maybe remove
                
                ## Lets remove the unsupervised
                proj_aug2_scrambled = proj_aug2[labels!=-1]
                proj_aug1_sup = proj_aug1[labels!=-1]
                
                proj_aug2_scrambled = scramble_according_labels(proj_aug2_scrambled, labels[labels!=-1])
                
                proj_concat_class = torch.cat([proj_aug1_sup, proj_aug2_scrambled], dim=0)
    
                
                if proj_aug2_scrambled.shape[0]>0:
                    # print('in class contrast')
                    #labels_concat = torch.cat([labels, labels], dim=0)
                    class_loss = self.class_contrast_loss(proj_concat_class, None, self.contrastive_config.n_views)
                    # labels_concat = torch.cat([labels, labels], dim=0)
                    # class_loss = self.class_contrast_loss(proj_concat, labels_concat, self.contrastive_config.n_views)
                    loss_dict['class_contrast_loss'] = class_loss.item()
                    total_loss += self.contrastive_config.class_contrast_weight * class_loss
            #else:
            #    loss_dict['class_contrast_loss'] = 0
                #total_loss += 0
        
        # 3. Reconstruction loss (from parent)
        if self.contrastive_config.recon_weight > 0:
            # print('hello recon')
            # print(values.shape)
            # print(outputs_og['recon_loss'])
            
            ## Could be less expensive because this is an extra fwd pass + GPU mem then
            # logits, recon_loss = super().forward(values, positions, pad_mask)

            # outputs_og = self.forward_cls(values, mask, positions, pad_mask, decode= (self.contrastive_config.recon_weight > 0))
            # cls_full = outputs_og['embeddings'][:, 0, :]
            # _, proj_orig, _ = self.extract_cls_and_project(cls_full)

            # TRYING TO AVOID AN EXTRA FORWARD PASS HERE, BUT THIS MEANS THE ACTUAL TOTAL MASK FOR RECONSTRUCTION IS THE COMBINATION 
            # OF THE MASKING FOR RECON AND MASKING FOR AUGMENTED? DO WE WANT TO KEEP THAT?
            #loss_dict['recon_loss'] = outputs_og['recon_loss'].item()
            loss_dict['recon_loss'] = outputs_aug1['recon_loss'].item()
            
            total_loss += self.contrastive_config.recon_weight * outputs_aug1['recon_loss'] #.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
            
        return {
            'logits': outputs_aug1['logits'],
            'loss': total_loss,
            'encoded': outputs_aug1['embeddings'], # not ideal i guess?
            'projections': proj_aug1,
            'aug1_projections': proj_aug1,
            'aug2_projections': proj_aug2,
            'loss_dict': loss_dict
        }

    
    def forward(self, values: torch.Tensor, mask: torch.Tensor,
                positions: torch.Tensor, pad_mask=None,
                label=None, *_, **__):

        #print("REMOVE THIS BEFORE RUNNING FOR REAL!!")
        #self.batch_counter+=1
        #print(self.batch_counter)
        #if self.batch_counter>1360:
        full_ = self.forward_fullinfo(values, mask, positions, pad_mask, label, self.compute_contrastive, self.decode)
        
        return full_['logits'], full_['loss']
        #else:
        #    return torch.tensor(0.0, device=values.device, requires_grad=True), torch.tensor(0.0, device=values.device, requires_grad=True)






class RoMAEPreTrainingContrastive(RoMAEForPreTraining):
    """
    Extends RoMAEForPreTraining to add contrastive learning; this implements the SupCon version for class-contrastive
    and does not have an augmented (e.g. noising) classical infoNCE contrastive part; use the other one for that

    This one also does not use a projection head, and applies (for now...let's see if it fails) the contrastive directly at the CLS level
    
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
       
        self.dim_use_cls = contrastive_config.cls_contrastive_dim if contrastive_config.cls_contrastive_dim else d_model
                
        # Contrastive loss
        self.class_contrast_loss = ContrastiveLoss(temperature=contrastive_config.temperature, 
                                                   unsup_in_denominator=contrastive_config.unsup_in_denominator)
        
        # Augmentation function
        self.augmentation_fn = augmentation_fn or (lambda x, y: x)

        self.compute_contrastive = True # 
        self.decode= contrastive_config.decode

        self.batch_counter=0


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
        
        if len(mask.shape)==1:
            mask = mask.unsqueeze(0)
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
                #
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
            cls_full: Full CLS token
        """
        # 
        
        # Split CLS if configured
        if self.contrastive_config.cls_contrastive_dim:
            cls_contrastive = cls_full[:, :self.contrastive_config.cls_contrastive_dim]
        else:
            cls_contrastive = cls_full
        
        # Project for contrastive learning
        # projections = self.projection_head(cls_contrastive)
        
        return cls_contrastive, cls_full
    
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
        Forward pass with contrastive learning (SupCon)
        
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
        #check if labels are not just unsup
        nolabelsup = False
        if labels is not None:
            # print('Labels')
            # print(torch.unique(labels))

            if labels[labels==-1].shape[0]==0:
                nolabelsup = True
                # print('no label sup')
            
        # if (not compute_contrastive): # | nolabelsup:
        #     # Act like normal RoMAEForPreTraining
        #     """ This needs fixing because it's not returning the same dict as the other branch """
        #     #print("ypassing contrastive loss -- this might break some stuff because it's not returning a dictionnary actually")
            
        #     #logits, recon_loss = super().forward(values, positions, pad_mask)
        #     return self.forward_cls(values, mask, positions, pad_mask, decode=True)


        ## With SupCon, we don't need to augment

        outputs_aug1 = self.forward_cls(values, mask, positions, pad_mask, decode=(self.contrastive_config.recon_weight > 0))
        proj_aug1, _ = self.extract_cls_and_project(outputs_aug1['embeddings'][:,0,:])
        
        
        loss_dict = {}
        total_loss = 0.0 #torch.Tensor([0.0]).to(values.device)
        
        # 2. Class-based contrastive loss
        if self.contrastive_config.class_contrast_weight > 0 and labels is not None:
            ## This assumes the use of K-in-batch trick , things might break if no two-positive
    
            class_loss = self.class_contrast_loss(proj_aug1, labels, 1)            
            loss_dict['class_contrast_loss'] = class_loss.item()
            total_loss += self.contrastive_config.class_contrast_weight * class_loss
            
        
        # 3. Reconstruction loss (from parent)
        if self.contrastive_config.recon_weight > 0:
            loss_dict['recon_loss'] = outputs_aug1['recon_loss'].item()
            
            total_loss += self.contrastive_config.recon_weight * outputs_aug1['recon_loss'] #.item()
        
        loss_dict['total_loss'] = total_loss.item()
        #print(total_loss)
        #print(total_loss)
        if wandb.run is not None:
            wandb.log({
                'class_contrast_loss':  loss_dict['class_contrast_loss'] if 'class_contrast_loss' in loss_dict else 0,  #class_loss.item(),
                'recon_loss': loss_dict['recon_loss'] if 'recon_loss' in loss_dict else 0 #.item(),
            }, commit=False)
            
        return {
            'logits': outputs_aug1['logits'],
            'loss': total_loss,
            'encoded': outputs_aug1['embeddings'], # CLS
            'projections': proj_aug1,  # A bit dumb / space waste here since we don't have a ProjectionHead anymore
            'loss_dict': loss_dict
        }


        
    def forward(self, values: torch.Tensor, mask: torch.Tensor,
                positions: torch.Tensor, pad_mask=None,
                label=None, *_, **__):

        #print("REMOVE THIS BEFORE RUNNING FOR REAL!!")
        #self.batch_counter+=1
        #print(self.batch_counter)
        #if self.batch_counter>1360:
        full_ = self.forward_fullinfo(values, mask, positions, pad_mask, label, self.compute_contrastive, self.decode)
        #print(full_['loss_dict'])
        return full_['logits'], full_['loss']
        #else:
        #    return torch.tensor(0.0, device=values.device, requires_grad=True), torch.tensor(0.0, device=values.device, requires_grad=True)



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