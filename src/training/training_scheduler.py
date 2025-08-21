import torch
from torch.optim import lr_scheduler
from typing import Dict, Any

class TrainingScheduler:
    """
    Learning rate scheduler for multi-stage training.
    """
    
    def __init__(self, optimizer, scheduler_config: Dict[str, Any]):
        self.optimizer = optimizer
        self.config = scheduler_config
        self.scheduler = self._create_scheduler()
    
    def _create_scheduler(self):
        """Create the appropriate scheduler."""
        scheduler_type = self.config.get('type', 'plateau')
        
        if scheduler_type == 'cosine':
            return lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('T_max', 100),
                eta_min=self.config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'plateau':
            return lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.config.get('mode', 'max'),
                factor=self.config.get('factor', 0.5),
                patience=self.config.get('patience', 5),
                min_lr=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            return lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        else:
            return None
    
    def step(self, metric=None):
        """Step the scheduler."""
        if self.scheduler:
            if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                if metric is not None:
                    self.scheduler.step(metric)
            else:
                self.scheduler.step()
    
    def get_last_lr(self):
        """Get last learning rate."""
        if hasattr(self.scheduler, 'get_last_lr'):
            return self.scheduler.get_last_lr()
        else:
            return [group['lr'] for group in self.optimizer.param_groups]