import torch
import torch.nn as  nn
import torch.nn.utils.prune as prune

class IndexedPruningMethod(prune.BasePruningMethod):
    """Prune all filters/featuremaps at given indices"""
    PRUNING_TYPE = 'structured'

    def __init__(self, indices=[],dim=-1):
        self.indices = indices
        self.dim = dim

    def compute_mask(self, t, default_mask):
        indices = self.indices
        mask = default_mask.clone()
        mask[indices, : ,: ,:] = 0
        return mask
    
