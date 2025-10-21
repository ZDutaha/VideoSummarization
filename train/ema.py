import math, torch

class EMA:
    def __init__(self, model, decay=0.999, device=None):
        self.decay = decay
        self.shadow = {}
        self.device = device
        for name, param in model.named_parameters():
            if param.requires_grad:
                v = param.detach().clone()
                if device is not None: v = v.to(device)
                self.shadow[name] = v

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new = param.detach()
                if self.device is not None: new = new.to(self.device)
                old = self.shadow[name]
                old.mul_(self.decay).add_(new, alpha=(1.0 - self.decay))

    def apply_to(self, model):
        self._backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}
