import torch

class BaseModel(torch.nn.Module):
    """ the template of creating a custom model"""
    def update(self):
        raise NotImplementedError

    def representation(self, x):
        raise NotImplementedError
    
    def forward(self, imgs, **kwargs):
        self.log = {}
        loss = 0
        return loss,self.log