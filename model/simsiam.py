"""
reference: https://github.com/facebookresearch/simsiam
# Note

We apply cosine decay for lr to predictor, whereas, removing it gains improvement.
The paper uses SGD to optimize the model. 

Warning: Not working with ViT.

# keypoints
- break symmetry
- Stop-gradient: the key to avoid collapse.
- Predictor: the key to converge. The loss remains high if removing the predictor. 2048-512-2048, no BN. The bottleneck instead of inverse bottleneck is vital for simsiam. The bottleneck prevents the predictor learning identity map, avoiding collapse.
- projector: Use BN at the end. 2048-2048-2048.
- BN: adding BN to the hidden layers is vital to the success of learning semantic representation. However, adding BN to the output of predictor will cause unstable training and the loss oscillates. 
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
import torch.nn.functional as F
from layers.backbone import create_backbone



@gin.configurable
class SimSiam(nn.Module):
    """warning: the model is not stable. The loss oscillates. """
    def __init__(self, 
                 proj_dim = 2048,
                 embed_dim=2048,
                 mlp_dim=512):
        super(SimSiam, self).__init__()
        backbone = create_backbone()
        self.embed_dim = embed_dim
        self.backbone = backbone
        
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True), # first layer
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True), # second layer
            nn.Linear(embed_dim, proj_dim,bias=False), # output layer
            nn.BatchNorm1d(proj_dim, affine=False),
        )
        

        
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, mlp_dim, bias=False),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(), 
            # nn.Softmax(), # simulate class centroid selection
            nn.Linear(mlp_dim, proj_dim)) # output layer        
        self.criterion = nn.CosineSimilarity(dim=1)

    @torch.no_grad()
    def representation(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        latent = self.backbone(x)
        proj = self.projector(latent)
        pred = self.predictor(proj)
        return dict(latent=latent,proj=proj,pred=pred)
    

    def forward(self, samples, **kwargs):
        self.log = {}
        x1,x2 = samples[:2]
        local_x = samples[2:]

        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        
        loss =  1 - (self.criterion(p1, z2.detach()).mean() + 
                     self.criterion(p2, z1.detach()).mean()) * 0.5

        loss_local = 0
        for lx in local_x:
            lz = self.backbone(lx)
            lp = self.predictor(lz)

            loss_local += 1 - (
                self.criterion(lp,z2.detach()) +
                self.criterion(lp,z1.detach()) 
            ).mean()/2

        self.log["loss"] = loss.item()
        self.log['qk@sim'] = self.criterion(p1.detach(), z1.detach()).mean().item()
        with torch.no_grad():
            
            self.log['qk@sim'] = F.cosine_similarity(p1,z1).mean().item()
            self.log['z@sim'] = F.cosine_similarity(p1,z2).mean().item()
        return loss, self.log
        