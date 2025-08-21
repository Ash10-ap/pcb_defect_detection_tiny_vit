#!/usr/bin/env python3
"""
Multi-stage training script for TinyViT-YOLOv8 PCB defect detection.

This script implements the three-stage training strategy:
1. Foundation: PKU-Market-PCB with frozen backbone
2. Transfer Learning: HRIPCB with progressive unfreezing  
3. Few-Shot: Novel defect adaptation

Usage:
    python scripts/train_multi_stage.py --config configs/tinivit_yolov8_pcb.yaml
    python scripts/train_multi_stage.py --config configs/tinivit_yolov8_pcb.yaml --stage transfer
    python scripts/train_multi_stage.py --resume experiments/checkpoints/stage_1_best.pt
"""

import argparse
import yaml
import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# Import with proper path handling
try:
    from src.training.multi_stage_trainer import MultiStageTrainer, get_default_stage_configs
    from src.datasets.pcb_dataset import PCBDataModule, get_dataset_configs
    from src.models.model_factory import create_model_from_config
    from src.utils.metrics import PCBMetrics
    from src.utils.export import export_all_formats
except ImportError:
    # Fallback for direct execution
    from training.multi_stage_trainer import MultiStageTrainer, get_default_stage_configs
    from datasets.pcb_dataset import PCBDataModule, get_dataset_configs
    from models.model_factory import create_model_from_config
    from utils.metrics import PCBMetrics
    from utils.export import export_all_formats

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Multi-stage TinyViT-YOLO Training')
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--stage', type=str, default='all',
                       choices=['all', 'foundation', 'transfer', 'few_shot'],
                       help='Training stage to run')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--data_root', type=str, default='data',
                       help='Root directory for datasets')
    parser.add_argument('--output_dir', type=str, default='experiments',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with limited data')
    
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_device(device_arg: str) -> torch.device:
    """Setup computation device."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print("Using CPU")
    else:
        device = torch.device(device_arg)
        print(f"Using device: {device}")
    
    return device

def setup_data_module(config: dict, args) -> PCBDataModule:
    """Setup data module for multi-stage training."""
    dataset_configs = get_dataset_configs()
    
    # Update dataset paths based on args
    for stage_config in dataset_configs.values():
        stage_config['dataset_root'] = os.path.join(
            args.data_root, 
            os.path.basename(stage_config['dataset_root'])
        )
    
    # Debug mode: limit samples
    if args.debug:
        for stage_config in dataset_configs.values():
            stage_config['max_train_samples'] = 100
            stage_config['max_val_samples'] = 50
    
    data_module = PCBDataModule(
        dataset_configs=dataset_configs,
        batch_size=args.batch_size or config['training']['batch_size'],
        num_workers=args.num_workers,
        img_size=config['model']['img_size']
    )
    
    return data_module

def train_single_stage(stage_name: str, config: dict, args, device: torch.device):
    """Train a single stage."""
    print(f"\n{'='*60}")
    print(f"TRAINING STAGE: {stage_name.upper()}")
    print(f"{'='*60}")
    
    # Setup data
    data_module = setup_data_module(config, args)
    data_module.setup(stage_name)
    
    # Create data loaders
    train_loader = data_module.train_dataloader(stage_name)
    val_loader = data_module.val_dataloader(stage_name)
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Setup trainer
    training_config = config['training'].copy()
    training_config['save_dir'] = os.path.join(args.output_dir, 'checkpoints')
    training_config['use_wandb'] = args.wandb
    training_config['experiment_name'] = f"{config['experiment']['name']}_{stage_name}"
    
    model_config = config['model'].copy()
    model_config['training_stage'] = stage_name
    
    trainer = MultiStageTrainer(
        config=training_config,
        model_config=model_config,
        device=device.type
    )
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Get stage configuration
    stage_configs = get_default_stage_configs()
    
    if stage_name == 'foundation':
        stage_config = stage_configs[0]
    elif stage_name == 'transfer':
        # Run all transfer stages
        transfer_configs = stage_configs[1:4]  # transfer_early, partial, full
        
        results = {}
        for i, stage_config in enumerate(transfer_configs):
            print(f"\nTransfer substage {i+1}: {stage_config['stage_name']}")
            
            stage_results = trainer.train_single_stage(
                stage_config, train_loader, val_loader, i
            )
            results[f'transfer_stage_{i+1}'] = stage_results
        
        return results
    
    elif stage_name == 'few_shot':
        # Setup few-shot specific configuration
        stage_config = {
            'stage_name': 'few_shot',
            'training_stage': 'few_shot',
            'epochs': 30,
            'optimizer': {
                'type': 'adam',
                'lr': 1e-5,
                'weight_decay': 1e-5
            },
            'scheduler': {
                'type': 'plateau',
                'factor': 0.5,
                'patience': 5,
                'min_lr': 1e-7
            },
            'patience': 10
        }
    else:
        raise ValueError(f"Unknown stage: {stage_name}")
    
    # Train the stage
    results = trainer.train_single_stage(
        stage_config, train_loader, val_loader, 0
    )
    
    # Save results
    results_path = os.path.join(args.output_dir, f'{stage_name}_results.json')
    trainer.save_training_results({stage_name: results}, results_path)
    
    return results

def train_all_stages(config: dict, args, device: torch.device):
    """Train all stages sequentially."""
    print("Starting multi-stage training pipeline...")
    
    stages = ['foundation', 'transfer', 'few_shot']
    all_results = {}
    
    for stage_name in stages:
        try:
            stage_results = train_single_stage(stage_name, config, args, device)
            all_results[stage_name] = stage_results
            
            print(f"\n✓ Stage '{stage_name}' completed successfully")
            
            # Update resume path for next stage
            args.resume = os.path.join(
                args.output_dir, 
                'checkpoints', 
                f'{stage_name}_best.pt'
            )
            
        except Exception as e:
            print(f"\n✗ Stage '{stage_name}' failed: {e}")
            break
    
    # Save final results
    final_results_path = os.path.join(args.output_dir, 'final_results.json')
    
    # Create a dummy trainer to save results
    training_config = config['training'].copy()
    training_config['save_dir'] = os.path.join(args.output_dir, 'checkpoints')
    
    trainer = MultiStageTrainer(
        config=training_config,
        model_config=config['model'],
        device=device.type
    )
    
    trainer.save_training_results(all_results, final_results_path)
    
    print(f"\n🎉 Multi-stage training completed!")
    print(f"Results saved to: {final_results_path}")
    
    return all_results

def export_final_model(config: dict, args, device: torch.device):
    """Export the final trained model to multiple formats."""
    print("\nExporting final model...")
    
    # Load best model
    best_model_path = os.path.join(args.output_dir, 'checkpoints', 'few_shot_best.pt')
    
    if not os.path.exists(best_model_path):
        print(f"Best model not found at: {best_model_path}")
        return
    
    # Create model
    model = create_model_from_config(config['model']['name'], config['model'])
    
    # Load weights
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Export to multiple formats
    example_input = torch.randn(1, 3, config['model']['img_size'], config['model']['img_size']).to(device)
    export_dir = os.path.join(args.output_dir, 'exported_models')
    
    exported_models = export_all_formats(
        model, example_input, export_dir, 'tinivit_yolo_final'
    )
    
    print("Exported models:")
    for format_name, path in exported_models.items():
        print(f"  {format_name}: {path}")

def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Setup device
    device = setup_device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config_save_path = os.path.join(args.output_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Configuration saved to: {config_save_path}")
    
    # Run training
    try:
        if args.stage == 'all':
            results = train_all_stages(config, args, device)
            
            # Export final model
            export_final_model(config, args, device)
            
        else:
            results = train_single_stage(args.stage, config, args, device)
        
        print("\n🎉 Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    # Set multiprocessing start method
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    
    main()