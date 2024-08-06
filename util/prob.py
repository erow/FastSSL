import torch
from torch import nn
import torch.distributed
import gin
import torch.nn.functional as F


class LinearProb(nn.Module):
    def __init__(self,data_path, names, representations_fn,num_classes=1):
        super().__init__()
        self.names = names
        self.representations_fn = representations_fn
        self.heads = nn.ModuleDict({
            name: nn.LazyLinear(num_classes) for name in names
        })
        self.regression = num_classes==1

        distributed = torch.distributed.is_initialized()
        self.optimizer = torch.optim.Adam(self.heads.parameters(), lr=1e-3)
        if num_classes>1:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()
        
        if ".ffcv" in data_path:
            from ffcv.loader import Loader, OrderOption
            from dataset.ffcv_transform import ValPipeline
            self.dl = Loader(data_path, batch_size=64, order=OrderOption.RANDOM, num_workers=10,drop_last=True, pipelines=ValPipeline(),distributed=distributed)
        else:
            from torchvision import datasets, transforms
            data = datasets.ImageFolder(data_path, 
                        transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
            self.dl = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True, num_workers=10,drop_last=True)
        self.next_batch = iter(self.dl)

    def step(self,x,y):
        if isinstance(x,list):
            x = x[0]
        self.step_train(x.cuda(), y.cuda())
        log = self.step_val()
        return log
    
    @torch.no_grad()
    def step_val(self):
        try:
            x, y = next(self.next_batch)
        except StopIteration:
            self.next_batch = iter(self.dl)
            x, y = next(self.next_batch)
        log = {}
        for name, z in zip(self.names, self.representations_fn(x,y)):
            pred = self.heads[name](z.detach())
            if self.regression:
                loss = self.criterion(pred.flatten(), y.float())
                log[name]=loss.item()
            else:
                acc = (pred.argmax(1) == y).float().mean()
                log[name] = acc.item()
        return log

    def step_train(self, x, y):
        self.optimizer.zero_grad()
        
        loss = 0
        for name, z in zip(self.names, self.representations_fn(x,y)):
            pred = self.heads[name](z.detach())
            if self.regression:
                loss = loss + self.criterion(pred.flatten(), y.float())

            else:
                loss = loss + self.criterion(pred, y)
                

        loss.backward()
        self.optimizer.step()
    
    

@gin.configurable(denylist=['model'])
def build_representations(model):
    if hasattr(model,'projector'):
        @torch.no_grad()
        def representations_fn(x,y):
            latent = model.representation(x)
            proj = model.projector(latent)
            return latent, proj
        names = "latent","proj"
    else:
        @torch.no_grad()
        def representations_fn(x,y):
            latent = model.representation(x)
            return latent,
        names = "latent",

    return names, representations_fn

@gin.configurable(denylist=['model'])
def build_representations_fn(model,fn=build_representations):
    return fn(model)    

