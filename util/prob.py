import torch
from torch import nn
import torch.distributed
import torch.distributed as dist
import gin
import torch.nn.functional as F
from ffcv.loader import Loader, OrderOption
from dataset.ffcv_transform import ValPipeline


class LinearProb(nn.Module):
    def __init__(self,data_path, names, representations_fn,num_classes=1000, device='cuda'):
        super().__init__()
        self.names = names
        self.representations_fn = representations_fn
        self.device = torch.device(device)
        self.heads = nn.ModuleDict({
            name: nn.LazyLinear(num_classes) for name in names
        })
        self.regression = num_classes==1

        distributed = torch.distributed.is_initialized()
        
        if num_classes>1:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()
        
        if ".ffcv" in data_path:
            self.dl = Loader(data_path, batch_size=64, order=OrderOption.RANDOM, num_workers=10,drop_last=True, pipelines=ValPipeline(device=device),distributed=distributed)
        else:
            from torchvision import datasets, transforms
            data = datasets.ImageFolder(data_path, 
                        transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
            distributed_sampler = torch.utils.data.distributed.DistributedSampler(data) if distributed else None
            self.dl = torch.utils.data.DataLoader(data, batch_size=64, shuffle=False, num_workers=4,drop_last=True, sampler=distributed_sampler)
        self.next_batch = iter(self.dl)
        
        # Move module to device
        self.to(self.device)
        # Initialize optimizer after moving to device
        self.optimizer = torch.optim.Adam(self.heads.parameters(), lr=1e-3)

    def step(self,x,y):
        if isinstance(x,list):
            x = x[0]
        # Ensure tensors are on the correct device
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        self.step_train(x, y)
        log = self.step_val()
        return log
    
    @torch.no_grad()
    def step_val(self):
        try:
            x, y = next(self.next_batch)
        except StopIteration:
            self.next_batch = iter(self.dl)
            x, y = next(self.next_batch)
        
        # Ensure tensors are on the correct device
        if not isinstance(x, torch.Tensor) or x.device != self.device:
            x = x.to(self.device, non_blocking=True) if isinstance(x, torch.Tensor) else torch.tensor(x, device=self.device)
        if not isinstance(y, torch.Tensor) or y.device != self.device:
            y = y.to(self.device, non_blocking=True) if isinstance(y, torch.Tensor) else torch.tensor(y, device=self.device)
            
        log = {}
        for name, z in self.representations_fn(x,y):
            pred = self.heads[name](z.detach())
            if self.regression:
                loss = self.criterion(pred.flatten(), y.float())
                # INSERT_YOUR_CODE
                # Collect metrics from all GPUs (processes) and report the mean
                if dist.is_available() and dist.is_initialized():
                    loss_tensor = loss.detach().clone()
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    loss_tensor /= dist.get_world_size()
                    loss = loss_tensor
                log[name]=loss.item()
            else:
                acc = (pred.argmax(1) == y).float().mean()
                # INSERT_YOUR_CODE
                # Collect metrics from all GPUs (processes) and report the mean
                if dist.is_available() and dist.is_initialized():
                    acc_tensor = acc.detach().clone()
                    dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
                    acc_tensor /= dist.get_world_size()
                    acc = acc_tensor
                log[name] = acc.item()
        return log

    def step_train(self, x, y):
        self.optimizer.zero_grad()
        
        loss = 0
        for name, z in self.representations_fn(x,y):
            pred = self.heads[name](z.detach())
            if self.regression:
                loss = loss + self.criterion(pred.flatten(), y.float())
            else:
                loss = loss + self.criterion(pred, y)
        loss.backward()
        self.optimizer.step()
    
    

@gin.configurable(denylist=['model'])
def build_representations(model):
    @torch.no_grad()
    def representations_fn(x,y):
        representation = model.representation(x)
        return representation.items()
    return model.representation_names, representations_fn

@gin.configurable(denylist=['model'])
def build_representations_fn(model,fn=build_representations):
    return fn(model)    

