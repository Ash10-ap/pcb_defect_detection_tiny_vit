import torch
import os
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Union
import logging

class CheckpointManager:
    """
    Manages model checkpoints during training.
    
    Features:
    - Save/load model checkpoints
    - Keep best N checkpoints
    - Automatic cleanup of old checkpoints
    - Metadata tracking
    """
    
    def __init__(self, 
                 save_dir: Union[str, Path],
                 max_checkpoints: int = 5,
                 monitor_metric: str = 'val_map',
                 mode: str = 'max'):
        """
        Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            monitor_metric: Metric to monitor for best checkpoint
            mode: 'max' or 'min' for best checkpoint selection
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.monitor_metric = monitor_metric
        self.mode = mode
        
        # Track checkpoints
        self.checkpoint_list = []
        self.best_score = float('-inf') if mode == 'max' else float('inf')
        self.best_checkpoint = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load existing checkpoints info
        self._load_checkpoint_info()
    
    def save_checkpoint(self,
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       metrics: Dict,
                       stage_name: str = None,
                       extra_info: Dict = None) -> str:
        """
        Save a training checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Training metrics
            stage_name: Optional training stage name
            extra_info: Additional information to save
            
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'stage_name': stage_name
        }
        
        if extra_info:
            checkpoint.update(extra_info)
        
        # Generate checkpoint filename
        if stage_name:
            filename = f'{stage_name}_epoch_{epoch:03d}.pt'
        else:
            filename = f'checkpoint_epoch_{epoch:03d}.pt'
        
        checkpoint_path = self.save_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Update checkpoint list
        checkpoint_info = {
            'path': str(checkpoint_path),
            'epoch': epoch,
            'metrics': metrics,
            'stage_name': stage_name
        }
        
        self.checkpoint_list.append(checkpoint_info)
        
        # Check if this is the best checkpoint
        metric_value = metrics.get(self.monitor_metric, 0)
        is_best = self._is_best_score(metric_value)
        
        if is_best:
            self.best_score = metric_value
            self.best_checkpoint = str(checkpoint_path)
            self._save_best_checkpoint(checkpoint_path)
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        # Save checkpoint info
        self._save_checkpoint_info()
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        if is_best:
            self.logger.info(f"New best checkpoint: {self.monitor_metric}={metric_value:.4f}")
        
        return str(checkpoint_path)
    
    def save_best_model(self,
                       model: torch.nn.Module,
                       score: float,
                       stage_name: str = None) -> str:
        """
        Save model as best checkpoint.
        
        Args:
            model: Model to save
            score: Current score
            stage_name: Optional stage name
            
        Returns:
            Path to best model
        """
        if self._is_best_score(score):
            self.best_score = score
            
            if stage_name:
                filename = f'{stage_name}_best.pt'
            else:
                filename = 'best_model.pt'
            
            best_path = self.save_dir / filename
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'score': score,
                'metric': self.monitor_metric,
                'stage_name': stage_name
            }
            
            torch.save(checkpoint, best_path)
            self.best_checkpoint = str(best_path)
            
            self.logger.info(f"Best model saved: {best_path} (score: {score:.4f})")
            
            return str(best_path)
        
        return self.best_checkpoint
    
    def load_checkpoint(self, 
                       checkpoint_path: Union[str, Path],
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer = None,
                       device: str = 'cpu') -> Dict:
        """
        Load checkpoint and restore model/optimizer state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            device: Device to load checkpoint on
            
        Returns:
            Checkpoint data
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        
        return checkpoint
    
    def load_best_checkpoint(self,
                           model: torch.nn.Module,
                           stage_name: str = None,
                           device: str = 'cpu') -> Dict:
        """
        Load the best checkpoint.
        
        Args:
            model: Model to load state into
            stage_name: Optional stage name to specify best checkpoint
            device: Device to load checkpoint on
            
        Returns:
            Checkpoint data
        """
        if stage_name:
            best_path = self.save_dir / f'{stage_name}_best.pt'
        else:
            best_path = self.save_dir / 'best_model.pt'
        
        if not best_path.exists() and self.best_checkpoint:
            best_path = Path(self.best_checkpoint)
        
        if not best_path.exists():
            raise FileNotFoundError("No best checkpoint found")
        
        return self.load_checkpoint(best_path, model, device=device)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the latest checkpoint."""
        if not self.checkpoint_list:
            return None
        
        latest = max(self.checkpoint_list, key=lambda x: x['epoch'])
        return latest['path']
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to the best checkpoint."""
        return self.best_checkpoint
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        return self.checkpoint_list.copy()
    
    def remove_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Remove a specific checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            
            # Remove from checkpoint list
            self.checkpoint_list = [
                cp for cp in self.checkpoint_list 
                if cp['path'] != str(checkpoint_path)
            ]
            
            self._save_checkpoint_info()
            self.logger.info(f"Removed checkpoint: {checkpoint_path}")
    
    def _is_best_score(self, score: float) -> bool:
        """Check if score is the best so far."""
        if self.mode == 'max':
            return score > self.best_score
        else:
            return score < self.best_score
    
    def _save_best_checkpoint(self, checkpoint_path: Path):
        """Create a copy of the best checkpoint."""
        best_path = self.save_dir / 'best_model.pt'
        shutil.copy2(checkpoint_path, best_path)
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit."""
        if len(self.checkpoint_list) <= self.max_checkpoints:
            return
        
        # Sort by epoch and keep the latest ones
        sorted_checkpoints = sorted(
            self.checkpoint_list, 
            key=lambda x: x['epoch'], 
            reverse=True
        )
        
        checkpoints_to_keep = sorted_checkpoints[:self.max_checkpoints]
        checkpoints_to_remove = sorted_checkpoints[self.max_checkpoints:]
        
        # Remove old checkpoint files
        for checkpoint_info in checkpoints_to_remove:
            checkpoint_path = Path(checkpoint_info['path'])
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                self.logger.info(f"Removed old checkpoint: {checkpoint_path}")
        
        # Update checkpoint list
        self.checkpoint_list = checkpoints_to_keep
    
    def _save_checkpoint_info(self):
        """Save checkpoint information to JSON file."""
        info_path = self.save_dir / 'checkpoint_info.json'
        
        info = {
            'checkpoint_list': self.checkpoint_list,
            'best_checkpoint': self.best_checkpoint,
            'best_score': self.best_score,
            'monitor_metric': self.monitor_metric,
            'mode': self.mode
        }
        
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
    
    def _load_checkpoint_info(self):
        """Load checkpoint information from JSON file."""
        info_path = self.save_dir / 'checkpoint_info.json'
        
        if info_path.exists():
            try:
                with open(info_path, 'r') as f:
                    info = json.load(f)
                
                self.checkpoint_list = info.get('checkpoint_list', [])
                self.best_checkpoint = info.get('best_checkpoint')
                self.best_score = info.get('best_score', 
                    float('-inf') if self.mode == 'max' else float('inf'))
                
                # Verify checkpoint files still exist
                valid_checkpoints = []
                for checkpoint_info in self.checkpoint_list:
                    if Path(checkpoint_info['path']).exists():
                        valid_checkpoints.append(checkpoint_info)
                
                self.checkpoint_list = valid_checkpoints
                
                # Verify best checkpoint exists
                if self.best_checkpoint and not Path(self.best_checkpoint).exists():
                    self.best_checkpoint = None
                    self.best_score = float('-inf') if self.mode == 'max' else float('inf')
                
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint info: {e}")
                self.checkpoint_list = []
                self.best_checkpoint = None


def test_checkpoint_manager():
    """Test checkpoint manager functionality."""
    print("Testing Checkpoint Manager")
    print("=" * 25)
    
    import tempfile
    import torch.nn as nn
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = CheckpointManager(temp_dir, max_checkpoints=3)
        
        # Create dummy model and optimizer
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Test saving checkpoints
        for epoch in range(5):
            metrics = {'val_map': 0.5 + epoch * 0.1, 'val_loss': 1.0 - epoch * 0.1}
            
            checkpoint_path = manager.save_checkpoint(
                model, optimizer, epoch, metrics, 'test_stage'
            )
            print(f"Saved checkpoint: {os.path.basename(checkpoint_path)}")
        
        # Test checkpoint list
        checkpoints = manager.list_checkpoints()
        print(f"Number of checkpoints: {len(checkpoints)}")
        print(f"Best checkpoint: {manager.get_best_checkpoint()}")
        
        # Test loading checkpoint
        latest_path = manager.get_latest_checkpoint()
        if latest_path:
            loaded_checkpoint = manager.load_checkpoint(latest_path, model, optimizer)
            print(f"Loaded checkpoint from epoch: {loaded_checkpoint['epoch']}")
        
        print("✓ Checkpoint manager test passed")


if __name__ == "__main__":
    test_checkpoint_manager()