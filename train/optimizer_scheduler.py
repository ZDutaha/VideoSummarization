from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
def build_optimizer(model, lr=2e-4, wd=1e-4): return AdamW(model.parameters(), lr=lr, weight_decay=wd)
def build_scheduler(optimizer): return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
