import torch
from torch import nn
from torchvision.models import resnet50, resnet18, resnet34, resnet101, resnet152
import torchvision.transforms as transforms
import gin

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def contrastive_loss(q, k,temperature=0.1):
    # normalize
    q = nn.functional.normalize(q, dim=1)
    k = nn.functional.normalize(k, dim=1)
    # gather all targets
    k = concat_all_gather(k)
    # Einstein sum is more intuitive
    logits = torch.einsum('nc,mc->nm', [q, k]) / temperature
    N = logits.shape[0]  # batch size per GPU
    labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
    return nn.CrossEntropyLoss()(logits, labels) * (2 * temperature)

@gin.configurable
def build_mlp(num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
    mlp = []
    for l in range(num_layers):
        dim1 = input_dim if l == 0 else mlp_dim
        dim2 = output_dim if l == num_layers - 1 else mlp_dim

        mlp.append(nn.Linear(dim1, dim2, bias=False))

        if l < num_layers - 1:
            mlp.append(nn.BatchNorm1d(dim2))
            mlp.append(nn.ReLU(inplace=True))
        elif last_bn:
            # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
            # for simplicity, we further removed gamma in BN
            mlp.append(nn.BatchNorm1d(dim2, affine=False))

    return nn.Sequential(*mlp)

import timm
@gin.configurable
class SimCLR(nn.Module):
    def __init__(self, backbone='resnet18', out_dim=2048,mlp_dim=512, temperature=0.5):
        super(SimCLR, self).__init__()
        self.temperature = temperature
        self.backbone = timm.create_model(backbone,pretrained=False,num_classes=out_dim)
        self.embed_dim = out_dim
        self.projector = build_mlp(2, out_dim, mlp_dim, out_dim)

    def representation(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        x = self.backbone(x)
        return x

    def forward(self, samples, **kwargs):
        x1,x2 = samples[:2]
        local_x = samples[2:]
        z1 = self.projector(self.representation(x1))
        z2 = self.projector(self.representation(x2))
        
        loss = contrastive_loss(z1,z2) + contrastive_loss(z2,z1)
        return loss, {}
        