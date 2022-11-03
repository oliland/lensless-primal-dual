import torch
from torch import nn


class EvaluationSystem(nn.Module):
    """
    We wish to keep our models separate from LightningModule to avoid an implicit dependency on pytorch-lightning.

    During training, our model is wrapped with a LightningModule:
        LightningModule(model=model)
    
    During evaluation, this class serves as an nn.Module with "model" as a top level key:
        nn.Module(model=model)
    
    This indirection allows us to recover our weights without needing to inherit from LightningModule.
    This is better suited for embedded devices which may not support Lightning for whatever reason,
    such as the Jetson Nano.
    """
    def __init__(self, model, checkpoint):
        super().__init__()
        self.model = model
        self.load_state_dict(checkpoint['state_dict'], strict=False)
        self.eval()

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            return self.model.forward(*args, **kwargs)

    @property
    def psfs(self):
        if hasattr(self.model, 'psfs'):
            return self.model.psfs
        else:
            return None
