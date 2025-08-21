"""SimDINO: Simplifying DINO via Coding Rate Regularization.
Reference: https://github.com/RobinWu218/SimDINO/tree/main

# note
- expa_type=1: z^2 = (z_student^2 + 2*z_student*z_teacher + z_teacher^2)/4
- ema momentum:  default=0.996, a higher value with small batches: for example use 0.9995 with batch size of 256.
- weight decay: a cosine schedule for WD in the origin. We use a fixed WD.
# training script
```
torchrun  main_pretrain_ema.py -m 0.9995 --data_set ffcv --data_path $FFCVTRAIN --batch_size 256 --epochs=100 --warmup_epochs=10  --ckpt_freq 10 --weight_decay 0.1 --blr 5e-4 --gin build_dataset.transform_fn=@MultiviewPipeline MultiviewPipeline.local_crops_number=10 build_model.model_fn=@SimDINO SimDINO.embed_dim=768 SimDINO.eps=0.05 
```

"""
import torch 
from torch import nn
import torch.distributed as dist
import torch.distributed.nn as dist_nn
import torch.nn.functional as F
import gin

from layers.backbone import create_backbone
from layers.operation import build_head, concat_all_gather
from timm.layers import trunc_normal_


def calc_compression( student_feat_list, teacher_feat_list):
    """
    Compute compression loss between student and teacher features.
    The average cosine similarity between the student and teacher features. This should be high.
    Arg:
        student_feat_list: [V1,B,C]
        teacher_feat_list: [V2,B,C]
    Return:
        sim: [V1,V2,B]
    """
    # Convert lists of tensors to a single tensor for vectorized operations
    
    sim = F.cosine_similarity(teacher_feat_list.unsqueeze(1), student_feat_list.unsqueeze(0), dim=-1)
    return sim

def code_rate_distortion(z1: torch.Tensor,z2: torch.Tensor , eps=1e-3):
    """
    Compute the rate distortion of a given tensor.
    """
    # normalization
    z1 = F.normalize(z1,dim=-1)
    z2 = F.normalize(z2,dim=-1)

    N = dist.get_world_size() if  dist.is_initialized() else 1
    m, p = z1.shape
    cov = z1.T.matmul(z2)
    I = torch.eye(p, device=cov.device)
    
    scalar = p / (m * N * eps)
    loss = torch.linalg.cholesky_ex(I + scalar * cov)[0].diagonal().log().sum()
    loss *=(p+N*m)/(p*N*m) 
    return loss


class SimHead(nn.Module):
    def __init__(self, in_dim, bottleneck_dim=256, use_bn=False, nlayers=3, hidden_dim=2048, ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        return x
    
@gin.configurable
class SimDINO(nn.Module):
    def __init__(self, 
                 embed_dim = 768,
                 out_dim=256, 
                 eps=0.5, coeff=1.0,
                 ):
        """
        dim: feature dimension (default: 256)
        """
        super().__init__()
        self.out_dim = out_dim
        
        self.eps = eps
        self.coeff = coeff

        self.embed_dim = embed_dim
        self.student = nn.Sequential(create_backbone(),
                                     SimHead(embed_dim,out_dim))
        
        _teacher = nn.Sequential(create_backbone(),
                                 SimHead(embed_dim,out_dim))
        _teacher.requires_grad_(False)
        self._teacher = _teacher
        self.update(0)
        self.num_samples = 256

    @torch.no_grad()
    def teacher(self,x):
        return self._teacher(x)
    
    @torch.no_grad()
    def update(self,m):
        for ema_v, model_v in zip(self._teacher.parameters(), self.student.parameters()):
            ema_v.data.mul_(m).add_((1 - m) * model_v.detach().data)

    def representation(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        latent = self._teacher[0](x)
        proj = self._teacher[1](latent)
        return dict(latent=latent,proj=proj)

    def forward(self, imgs, **kwargs):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """
        self.log = {}
        x1, x2 = imgs[:2]
        local_x = imgs[2:]

        # get the embeddings
        student_list= torch.stack([self.student(x) for x in imgs])
        with torch.no_grad():
            teacher_list = torch.stack([self.teacher(x1),self.teacher(x2)])
        
        V,B,C = student_list.shape
        # calc_compression    
        global_sim = (
            F.cosine_similarity(student_list[0],teacher_list[1],dim=-1)+
            F.cosine_similarity(student_list[1],teacher_list[0],dim=-1)).mean()/2
        if V>2:
            local_sim = calc_compression(student_list[2:],teacher_list)
            local_sim = local_sim.mean()
            self.log['local_sim'] = local_sim.item()
        
        
        
        # calc_expansion
        cr = 0
        for v in range(2):
            z1 = student_list[v] 
            z2 = teacher_list[v] 
            z = (z1+z2)/2
            cr = cr + code_rate_distortion(z, (z),self.eps)
        
        cr /= 2
        loss = - (cr+ self.coeff*(local_sim+global_sim))
        
        self.log['sim'] = global_sim.item()
        self.log['cr']=cr.item()
        return loss,self.log

if __name__ == '__main__':
    model = SimDINO(out_dim=32)
    x = torch.rand(10,3,32,32)
    print(model([x,x]))
    
