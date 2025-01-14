import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed import group, ReduceOp, is_initialized
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch import Tensor


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not is_initialized():
        return tensor
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def contrastive_loss(q, k,temperature=0.1):
    # NT-Xent (the normalized temperature-scaled cross entropy loss), applied in [Improved Deep Metric Learning with Multi-class N-pair Loss Objective]
    # normalize
    q = nn.functional.normalize(q, dim=1)
    k = nn.functional.normalize(k, dim=1)
    # gather all targets
    k = concat_all_gather(k)
    # Einstein sum is more intuitive
    logits = torch.einsum('nc,mc->nm', [q, k]) / temperature
    N = logits.shape[0]  # batch size per GPU
    rank = torch.distributed.get_rank() if is_initialized() else 0
    labels = (torch.arange(N, dtype=torch.long) + N * rank).to(logits.device)
    return nn.CrossEntropyLoss()(logits, labels)


class AllGatherGrad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        tensor: Tensor,
        group: Optional["torch.distributed.ProcessGroup"] = group.WORLD,
    ) -> Tensor:
        ctx.group = group

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor, group=group)
        gathered_tensor = torch.stack(gathered_tensor, dim=0)

        return gathered_tensor

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[Tensor, None]:
        # print("backward------------->")
        # print(grad_output)
        grad_output = torch.cat(grad_output)

        torch.distributed.all_reduce(grad_output, op=torch.distributed.ReduceOp.SUM, async_op=False, group=ctx.group)

        return grad_output[torch.distributed.get_rank()], None  

def concat_all_gather_grad(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not is_initialized():
        return tensor
    return AllGatherGrad.apply(tensor).flatten(0,1)

def build_head(num_layers, input_dim, mlp_dim, output_dim, hidden_bn=True,activation=nn.ReLU,
               last_norm='bn',):
    mlp = []
    for l in range(num_layers):
        dim1 = input_dim if l == 0 else mlp_dim
        dim2 = output_dim if l == num_layers - 1 else mlp_dim

        if l == num_layers-1:
            mlp.append(nn.Linear(dim1, dim2, bias=False))
        else:
            mlp.append(nn.Linear(dim1, dim2, bias=True))

        if l < num_layers - 1:
            if hidden_bn:
                mlp.append(nn.BatchNorm1d(dim2))
            mlp.append(activation())
        else:
            if last_norm=='bn':
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))
            elif last_norm=='ln':
                mlp.append(nn.LayerNorm(dim2))
            elif last_norm=='none':
                pass
            else:
                raise NotImplementedError(f"last_norm={last_norm} not implemented")

    return nn.Sequential(*mlp)


def patchify(imgs,p=16):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    return x

def unpatchify(x,p=16):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs

def pixel_norm(target):
    """
    target: (N, L, C)
    """
    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True)
    target = (target - mean) / (var + 1.e-6)**.5
    return target

