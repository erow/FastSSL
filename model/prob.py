import torch
class OnlineProb():
    def __init__(self, model,lr=1e-3):
        self.model = model
        from torch import nn
        embd_dim = model.embed_dim
        self.classifer = nn.Sequential(
            nn.BatchNorm1d(embd_dim),
            nn.Linear(embd_dim, 1000),
        )
        self.optimizer = torch.optim.Adam(self.classifer.parameters(), lr=lr,)
        
    def train_one_step(self, samples, targets):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                features = self.model.representation(samples)
        logits = self.classifer(features)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return acc.item()
    