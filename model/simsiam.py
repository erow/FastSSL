"""
reference: https://github.com/facebookresearch/simsiam
# Note

We apply cosine decay for lr to predictor, whereas, removing it gains improvement.
The paper uses SGD to optimize the model. 


# keypoints
- break symmetry
- Stop-gradient: the key to avoid collapse.
- Predictor: the key to converge. The loss remains high if removing the predictor. 2048-512-2048, no BN. The bottleneck instead of inverse bottleneck is vital for simsiam.
- projector: Use BN at the end. 2048-2048-2048.
- BN: dding BN to the hidden layers is vital to the success of learning semantic representation. However, adding BN to the output of predictor will cause unstable training and the loss oscillates. 
- Hypothesis: The presence of stop-gradient is the consequence of introducing the extra set of variables.


# Result:

| Model    | Note      | pre-train epochs | batch size | linprob (top-1)|
|----------|-----------|:----------------:|:----------:|:----------:|
| resnet50 | official  | 100              | 512        | 68.1       |
| resnet50 | official  | 100              | 256        | 68.3       |
| resnet50 | our impl. | 100              | 256        |            |
"""
import torch
from torch import nn
import gin
import timm
from .operation import *


@gin.configurable
class SimSiam(nn.Module):
    def __init__(self, backbone='resnet50', 
                 out_dim=2048,
                 hidden_dim=2048,
                 mlp_dim=512):
        super(SimSiam, self).__init__()
        self.embed_dim = out_dim
        backbone = timm.create_model(backbone, num_classes=hidden_dim)
        projector = build_mlp(2,hidden_dim,hidden_dim,out_dim)
        self.backbone = nn.Sequential(backbone,
                                      nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                      projector)
        self.predictor = build_mlp(2, out_dim, mlp_dim, out_dim, False)

        self.criterion = nn.CosineSimilarity(dim=1)

    def representation(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        x = self.backbone(x)
        return x
    
    def update(self):
        pass

    def forward(self, samples, **kwargs):
        self.log = {}
        x1,x2 = samples[:2]
        local_x = samples[2:]

        z1 = self.backbone(x1)
        p1 = self.predictor(z1)

        z2 = self.backbone(x2)
        p2 = self.predictor(z2)
        
        loss = 1 - (self.criterion(p1, z2.detach()).mean() + 
                     self.criterion(p2, z1.detach()).mean()) * 0.5

        loss_local = 0
        for lx in local_x:
            lz = self.backbone(lx)
            lp = self.predictor(lz)

            loss_local += 1 - (
                self.criterion(lp,z2.detach()) +
                self.criterion(lp,z1.detach()) 
            ).mean()/2

        self.log = {
            "loss":loss.item(),
            "loss_local":loss_local if loss_local==0 else loss_local.item() 
        }
        return loss + loss_local, self.log
        