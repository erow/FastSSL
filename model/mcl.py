"""SimDINO: Simplifying DINO via Coding Rate Regularization.
Reference: https://github.com/RobinWu218/SimDINO/tree/main

torchrun main_pretrain_ema.py --data_set cifar10 --data_path ../data/torch_data/  --batch_size 512 --epochs=200 --warmup_epochs=10 --ckpt_freq 100  --cfgs configs/cifar.gin configs/vitt.gin --gin build_model.model_fn=@SimDINO SimDINO.embed_dim=192 SimDINO.out_dim=128  MCRLoss.coeff=100 -m 0.996 --output_dir outputs/simdino_cifar10
"""
import torch 
from torch import nn
import torch.distributed as dist
import torch.distributed.nn as dist_nn
import torch.nn.functional as F
import gin

from layers.backbone import create_backbone
from model.dino import DINOHead


@gin.configurable
class MCRLoss(nn.Module):
    def __init__(self, ncrops=2, reduce_cov=0, expa_type=1, eps=0.5, coeff=1.0):
        """
        Args:
            ncrops (int, optional): _description_. Defaults to 2.
            reduce_cov (int, optional): Whether or not all_reduce covariance matrices across gpus. Defaults to 0.
            expa_type (int, optional): Whether or not apply smoothing in expansion_term. Defaults to 1.
            eps (float, optional): eps for TCR. Defaults to 0.5.
            coeff (float, optional): coefficient of cosine similarity. Defaults to 1.0.
        """
        super().__init__()
        self.ncrops = ncrops
        self.eps = eps
        self.coeff = coeff
        self.reduce_cov = reduce_cov
        self.expa_type = expa_type

    def forward(self, student_feat, teacher_feat):
        """
        Expansion Loss and Compression Loss between features of the teacher and student networks.
        """
        student_feat = student_feat.view(self.ncrops, -1, student_feat.shape[-1])
        teacher_feat = teacher_feat.view(2, -1, teacher_feat.shape[-1])
        if student_feat.isnan().any():
            print("Warning: NaN student_feat")
            raise ValueError("NaN loss")
        
        comp_loss = self.calc_compression(student_feat, teacher_feat)
        if self.expa_type == 0: # only compute expansion on global views
            expa_loss = self.calc_expansion(student_feat[:len(teacher_feat)])
        elif self.expa_type == 1:
            expa_loss = self.calc_expansion((student_feat[:2]+teacher_feat)/2)
        loss = - self.coeff * comp_loss - expa_loss
        return loss, comp_loss.detach(), expa_loss.detach()
    
    def calc_compression(self, student_feat_list, teacher_feat_list):
        """
        Compute compression loss between student and teacher features.
        The average cosine similarity between the student and teacher features. This should be high.
        """
        # Convert lists of tensors to a single tensor for vectorized operations
        
        sim = F.cosine_similarity(teacher_feat_list.unsqueeze(1), student_feat_list.unsqueeze(0), dim=-1)
        sim.view(-1, sim.shape[-1])[:: (len(student_feat_list) + 1), :].fill_(0)  # Trick to fill diagonal
        
        n_loss_terms = len(teacher_feat_list)* len(student_feat_list) - min(len(teacher_feat_list), len(student_feat_list))
        # Sum the cosine similarities
        comp_loss = sim.mean(2).sum()/n_loss_terms
        # global_comp_loss = (sim[:, :len(teacher_feat_list)].mean(2).sum()).detach_().div_(len(teacher_feat_list))
        
        if torch.isnan(comp_loss):
            print("Warning: NaN comp_loss")
            raise ValueError("NaN loss")
        return comp_loss
    
    def calc_expansion(self, feat_list) -> torch.Tensor:
        """
        Compute expansion loss using Coding Rate estimation.
        This denotes the information content of the features. This should be high.
        """
        cov_list = []
        num_views = len(feat_list)
        m, p = feat_list[0].shape
        
        cov_list = [W.T.matmul(W) for W in feat_list]
        cov_list = torch.stack(cov_list)
        N=1
        if dist.is_initialized():
            N = dist.get_world_size()
            if self.reduce_cov == 1:
                cov_list = dist_nn.all_reduce(cov_list)
        scalar = p / (m * N * self.eps)
        I = torch.eye(p, device=cov_list[0].device)
        loss:torch.Tensor = 0
        for i in range(num_views):
            lossi = torch.linalg.cholesky_ex(I + scalar * cov_list[i])[0].diagonal().log().sum()
            if torch.isnan(lossi):
                print("Warning: NaN comp_loss")
                torch.save(feat_list, "z.pt")
                raise ValueError("NaN loss")
            loss += lossi
        loss /= num_views
        # loss *= (p+N*m)/(p*N*m) # the balancing factor gamma, you can also use the next line. This is ultimately a heuristic, so feel free to experiment.
        # loss *= ((self.eps * N * m) ** 0.5 / p)
        return loss

def rate_distortion(z,eps=1e-3):
    """
    Compute the rate distortion of a given tensor.
    """
    m, p = z.shape
    cov = z.T.matmul(z)
    I = torch.eye(p, device=cov.device)
    scalar = p / (m * eps)
    return torch.linalg.cholesky_ex(I + scalar * cov)[0].diagonal().log().sum()

@gin.configurable
class SimDINO(nn.Module):
    def __init__(self, 
                 embed_dim = 2048,
                 out_dim=60000, 
                 norm_last_layer = True,
                 ):
        """
        dim: feature dimension (default: 60000)
        teacher_temp: softmax temperature for teacher. Final value (after linear warmup) of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend  starting with the default value of 0.04 and increase this slightly if needed.
        student_temp: 
        """
        super().__init__()
        self.out_dim = out_dim

        # build encoders
        self.loss_fn = MCRLoss()
        
        self.embed_dim = embed_dim
        self.student = nn.Sequential(create_backbone(),
                                     DINOHead(embed_dim, out_dim,norm_last_layer=norm_last_layer))
        
        _teacher = nn.Sequential(create_backbone(),
                                 DINOHead(embed_dim, out_dim))
        _teacher.requires_grad_(False)
        self._teacher = _teacher
        self.update(0)

    @torch.no_grad()
    def teacher(self,x):
        return self._teacher(x)
    
    @torch.no_grad()
    def update(self,m):
        for ema_v, model_v in zip(self._teacher.state_dict().values(), self.student.state_dict().values()):
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

        # predict the distribution of clusters
        student_output= torch.cat([self.student(x1),self.student(x2)])
        with torch.no_grad():
            teacher_output = torch.cat([self.teacher(x1),self.teacher(x2)])
        
        loss, comp_loss, expa_loss = self.loss_fn(student_output, teacher_output)
        
        self.log['comp_loss'] = comp_loss.item()
        self.log['expa_loss'] = expa_loss.item()
        
        return loss,self.log

if __name__ == '__main__':
    model = SimDINO(out_dim=32)
    x = torch.rand(10,3,32,32)
    print(model([x,x]))
    