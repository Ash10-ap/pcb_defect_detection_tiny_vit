import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import time
from typing import Dict, List, Optional, Tuple
import wandb
from tqdm import tqdm

try:
    from ..models.model_factory import create_model_from_config
    from .training_scheduler import TrainingScheduler
    from .loss_functions import CombinedLoss
    from ..utils.metrics import PCBMetrics
    from ..utils.checkpoint import CheckpointManager
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from models.model_factory import create_model_from_config
    from training.training_scheduler import TrainingScheduler
    from training.loss_functions import CombinedLoss
    from utils.metrics import PCBMetrics
    from utils.checkpoint import CheckpointManager

class MultiStageTrainer:
    """
    Multi-stage trainer for TinyViT-YOLOv8 PCB defect detection.
    
    Implements the three-stage training strategy:
    1. Foundation: PKU-Market-PCB with frozen backbone
    2. Transfer Learning: HRIPCB with progressive unfreezing
    3. Few-Shot: Novel defect adaptation
    """
    
    def __init__(self, 
                 config: Dict,
                 model_config: Dict,
                 device: str = 'cuda'):
        self.config = config
        self.model_config = model_config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = create_model_from_config(
            config['model_name'], 
            model_config
        ).to(self.device)
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = CombinedLoss(config['loss_config'])
        self.metrics = PCBMetrics(config['num_classes'])
        self.checkpoint_manager = CheckpointManager(config['save_dir'])
        
        # Training state
        self.current_stage = 0
        self.best_map = 0.0
        self.training_history = []
        
        # Setup wandb logging if enabled
        if config.get('use_wandb', False):
            wandb.init(
                project=config['project_name'],
                name=config['experiment_name'],
                config=config
            )
    
    def train_all_stages(self, 
                        stage_configs: List[Dict],
                        train_loaders: List[DataLoader],
                        val_loaders: List[DataLoader]) -> Dict:
        """
        Train all stages sequentially.
        
        Args:
            stage_configs: List of configuration for each stage
            train_loaders: List of training data loaders for each stage
            val_loaders: List of validation data loaders for each stage
            
        Returns:
            Training history and final metrics
        """
        results = {}
        
        for stage_idx, (stage_config, train_loader, val_loader) in enumerate(
            zip(stage_configs, train_loaders, val_loaders)
        ):
            print(f"\n{'='*50}")
            print(f"Starting Stage {stage_idx + 1}: {stage_config['stage_name']}")
            print(f"{'='*50}")
            
            # Update model training stage
            self.model.set_training_stage(stage_config['training_stage'])
            
            # Train single stage
            stage_results = self.train_single_stage(
                stage_config, train_loader, val_loader, stage_idx
            )
            
            results[f'stage_{stage_idx + 1}'] = stage_results
            
            # Save stage checkpoint
            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, stage_idx, 
                stage_results['best_map'], stage_config['stage_name']
            )
        
        return results
    
    def train_single_stage(self,
                          stage_config: Dict,
                          train_loader: DataLoader,
                          val_loader: DataLoader,
                          stage_idx: int) -> Dict:
        """Train a single stage."""
        self.current_stage = stage_idx
        
        # Setup optimizer and scheduler for this stage
        self._setup_optimizer_scheduler(stage_config)
        
        # Training loop
        stage_history = []
        best_map = 0.0
        patience_counter = 0
        
        for epoch in range(stage_config['epochs']):
            print(f"\nEpoch {epoch + 1}/{stage_config['epochs']}")
            
            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            val_metrics = self._validate_epoch(val_loader, epoch)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step(val_metrics['map'])
            
            # Log metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'stage': stage_idx + 1,
                'train': train_metrics,
                'val': val_metrics,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            
            stage_history.append(epoch_metrics)
            self.training_history.append(epoch_metrics)
            
            # Wandb logging
            if self.config.get('use_wandb', False):
                wandb.log({
                    f'stage_{stage_idx + 1}/train_loss': train_metrics['loss'],
                    f'stage_{stage_idx + 1}/val_map': val_metrics['map'],
                    f'stage_{stage_idx + 1}/val_map50': val_metrics['map50'],
                    f'stage_{stage_idx + 1}/lr': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch + 1
                })
            
            # Early stopping and best model saving
            if val_metrics['map'] > best_map:
                best_map = val_metrics['map']
                patience_counter = 0
                
                # Save best model
                self.checkpoint_manager.save_best_model(
                    self.model, val_metrics['map'], stage_idx
                )
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= stage_config.get('patience', 10):
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Print progress
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Val mAP: {val_metrics['map']:.4f}, "
                  f"Val mAP@50: {val_metrics['map50']:.4f}")
        
        return {
            'history': stage_history,
            'best_map': best_map,
            'final_metrics': epoch_metrics
        }
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f"Training")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Compute loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        
        return {
            'loss': avg_loss,
            'num_batches': num_batches
        }
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict:
        """Validate for one epoch."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            
            for images, targets in pbar:
                images = images.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                outputs = self.model(images)
                
                # Post-process predictions
                predictions = self._post_process_predictions(outputs)
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Compute metrics
        metrics = self.metrics.compute_metrics(all_predictions, all_targets)
        
        return metrics
    
    def _post_process_predictions(self, outputs):
        """Post-process model outputs to get final predictions."""
        # This would typically involve NMS and coordinate conversion
        # For now, return a placeholder
        predictions = []
        
        for output in outputs:
            # Placeholder post-processing
            pred = {
                'boxes': output[..., :4],
                'scores': output[..., 4],
                'labels': output[..., 5:].argmax(dim=-1)
            }
            predictions.append(pred)
        
        return predictions
    
    def _setup_optimizer_scheduler(self, stage_config: Dict):
        """Setup optimizer and scheduler for current stage."""
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Optimizer
        optimizer_config = stage_config['optimizer']
        
        if optimizer_config['type'] == 'adam':
            self.optimizer = optim.Adam(
                trainable_params,
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        elif optimizer_config['type'] == 'adamw':
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        elif optimizer_config['type'] == 'sgd':
            self.optimizer = optim.SGD(
                trainable_params,
                lr=optimizer_config['lr'],
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        
        # Scheduler
        scheduler_config = stage_config.get('scheduler', {})
        
        if scheduler_config.get('type') == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=stage_config['epochs'],
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_config.get('type') == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5),
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
        else:
            self.scheduler = None
    
    def save_training_results(self, results: Dict, save_path: str):
        """Save training results to file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Training results saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        
        return checkpoint


def get_default_stage_configs():
    """Get default configurations for all three training stages."""
    return [
        {
            'stage_name': 'foundation',
            'training_stage': 'foundation',
            'epochs': 80,
            'optimizer': {
                'type': 'adam',
                'lr': 1e-3,
                'weight_decay': 1e-4
            },
            'scheduler': {
                'type': 'cosine',
                'min_lr': 1e-6
            },
            'patience': 10
        },
        {
            'stage_name': 'transfer_early',
            'training_stage': 'transfer_early',
            'epochs': 20,
            'optimizer': {
                'type': 'adam',
                'lr': 5e-4,
                'weight_decay': 1e-4
            },
            'scheduler': {
                'type': 'plateau',
                'factor': 0.5,
                'patience': 5,
                'min_lr': 1e-6
            },
            'patience': 8
        },
        {
            'stage_name': 'transfer_partial',
            'training_stage': 'transfer_partial',
            'epochs': 20,
            'optimizer': {
                'type': 'adam',
                'lr': 2e-4,
                'weight_decay': 1e-4
            },
            'scheduler': {
                'type': 'plateau',
                'factor': 0.5,
                'patience': 5,
                'min_lr': 1e-6
            },
            'patience': 8
        },
        {
            'stage_name': 'transfer_full',
            'training_stage': 'transfer_full',
            'epochs': 20,
            'optimizer': {
                'type': 'adam',
                'lr': 1e-4,
                'weight_decay': 1e-4
            },
            'scheduler': {
                'type': 'plateau',
                'factor': 0.5,
                'patience': 5,
                'min_lr': 1e-6
            },
            'patience': 8
        }
    ]


def test_multi_stage_trainer():
    """Test the multi-stage trainer."""
    print("Testing Multi-Stage Trainer")
    print("=" * 30)
    
    # Mock configuration
    config = {
        'model_name': 'tinivit_yolo_small',
        'num_classes': 6,
        'save_dir': 'experiments/test',
        'use_wandb': False,
        'loss_config': {'bbox_weight': 1.0, 'cls_weight': 1.0},
        'project_name': 'test',
        'experiment_name': 'test_run'
    }
    
    model_config = {
        'training_stage': 'foundation'
    }
    
    trainer = MultiStageTrainer(config, model_config, device='cpu')
    
    print(f"Trainer initialized with device: {trainer.device}")
    print(f"Model type: {type(trainer.model).__name__}")
    
    print("Multi-stage trainer test passed!")


if __name__ == "__main__":
    test_multi_stage_trainer()