import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import timm

try:
    from .tinivit_backbone import TinyViTBackbone
    from .yolo_integration import TinyViTYOLO, create_tinivit_yolo
    from ..attention.attention_fusion import AttentionFusion
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from models.tinivit_backbone import TinyViTBackbone
    from models.yolo_integration import TinyViTYOLO, create_tinivit_yolo
    from attention.attention_fusion import AttentionFusion

class ModelFactory:
    """
    Factory class for creating different model configurations for
    PCB defect detection experiments.
    """
    
    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any]) -> nn.Module:
        """
        Create a model based on type and configuration.
        
        Args:
            model_type: Type of model to create
            config: Model configuration dictionary
            
        Returns:
            Initialized model
        """
        if model_type == 'tinivit_yolo':
            return ModelFactory._create_tinivit_yolo(config)
        elif model_type == 'baseline_yolo':
            return ModelFactory._create_baseline_yolo(config)
        elif model_type == 'tinivit_backbone_only':
            return ModelFactory._create_tinivit_backbone(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def _create_tinivit_yolo(config: Dict[str, Any]) -> TinyViTYOLO:
        """Create TinyViT-YOLO model."""
        backbone_config = config.get('backbone_config', {
            'model_name': 'tiny_vit_21m_224',
            'pretrained': True,
            'img_size': 640
        })
        
        attention_config = config.get('attention_config', {
            'attention_type': 'cbam',
            'use_cross_scale': True,
            'use_residual': True
        })
        
        yolo_config = config.get('yolo_config', {
            'nc': 6,
            'depth_multiple': 0.33,
            'width_multiple': 0.25
        })
        
        training_stage = config.get('training_stage', 'foundation')
        
        return TinyViTYOLO(
            backbone_config=backbone_config,
            attention_config=attention_config,
            yolo_config=yolo_config,
            training_stage=training_stage
        )
    
    @staticmethod
    def _create_baseline_yolo(config: Dict[str, Any]) -> nn.Module:
        """Create baseline YOLOv8 model for comparison."""
        from ultralytics import YOLO
        
        model_size = config.get('model_size', 'n')  # n, s, m, l, x
        num_classes = config.get('nc', 6)
        
        # Load pretrained YOLOv8 model
        model = YOLO(f'yolov8{model_size}.pt')
        
        # Modify for custom number of classes if needed
        if num_classes != 80:  # COCO has 80 classes
            model.model[-1].nc = num_classes
            model.model[-1].anchors = model.model[-1].anchors.clone()
        
        return model
    
    @staticmethod
    def _create_tinivit_backbone(config: Dict[str, Any]) -> TinyViTBackbone:
        """Create TinyViT backbone only."""
        return TinyViTBackbone(**config)

def get_model_configs():
    """
    Get predefined model configurations for different experimental setups.
    
    Returns:
        Dictionary of model configurations
    """
    configs = {
        'tinivit_yolo_small': {
            'backbone_config': {
                'model_name': 'tiny_vit_11m_224',
                'pretrained': True,
                'img_size': 640
            },
            'attention_config': {
                'attention_type': 'cbam',
                'use_cross_scale': False,
                'use_residual': True
            },
            'yolo_config': {
                'nc': 6,
                'depth_multiple': 0.33,
                'width_multiple': 0.25
            }
        },
        
        'tinivit_yolo_medium': {
            'backbone_config': {
                'model_name': 'tiny_vit_21m_224',
                'pretrained': True,
                'img_size': 640
            },
            'attention_config': {
                'attention_type': 'cbam',
                'use_cross_scale': True,
                'use_residual': True
            },
            'yolo_config': {
                'nc': 6,
                'depth_multiple': 0.5,
                'width_multiple': 0.5
            }
        },
        
        'tinivit_yolo_eca': {
            'backbone_config': {
                'model_name': 'tiny_vit_21m_224',
                'pretrained': True,
                'img_size': 640
            },
            'attention_config': {
                'attention_type': 'eca_spatial',
                'use_cross_scale': True,
                'use_residual': True
            },
            'yolo_config': {
                'nc': 6,
                'depth_multiple': 0.33,
                'width_multiple': 0.25
            }
        },
        
        'tinivit_yolo_enhanced': {
            'backbone_config': {
                'model_name': 'tiny_vit_21m_224',
                'pretrained': True,
                'img_size': 640
            },
            'attention_config': {
                'attention_type': 'enhanced_cbam',
                'use_cross_scale': True,
                'use_residual': True
            },
            'yolo_config': {
                'nc': 6,
                'depth_multiple': 0.5,
                'width_multiple': 0.5
            }
        },
        
        'baseline_yolo_n': {
            'model_size': 'n',
            'nc': 6
        },
        
        'baseline_yolo_s': {
            'model_size': 's',
            'nc': 6
        }
    }
    
    return configs

def create_model_from_config(config_name: str, 
                           custom_config: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    Create a model from a predefined configuration.
    
    Args:
        config_name: Name of the predefined configuration
        custom_config: Optional custom configuration to override defaults
        
    Returns:
        Initialized model
    """
    configs = get_model_configs()
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    config = configs[config_name].copy()
    
    # Override with custom configuration if provided
    if custom_config:
        config.update(custom_config)
    
    # Determine model type based on config name
    if 'tinivit_yolo' in config_name:
        model_type = 'tinivit_yolo'
    elif 'baseline_yolo' in config_name:
        model_type = 'baseline_yolo'
    else:
        model_type = 'tinivit_yolo'  # Default
    
    return ModelFactory.create_model(model_type, config)

def compare_model_sizes():
    """
    Compare model sizes and parameter counts for different configurations.
    """
    configs = get_model_configs()
    
    print("Model Size Comparison")
    print("=" * 50)
    
    for config_name in configs:
        try:
            model = create_model_from_config(config_name)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Estimate model size in MB (assuming float32)
            model_size_mb = total_params * 4 / (1024 * 1024)
            
            print(f"\n{config_name}:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Model size: {model_size_mb:.2f} MB")
            
        except Exception as e:
            print(f"\n{config_name}: Error - {e}")

def test_model_factory():
    """Test the model factory functionality."""
    print("Testing Model Factory")
    print("=" * 30)
    
    # Test TinyViT-YOLO creation
    config = {
        'backbone_config': {'model_name': 'tiny_vit_21m_224', 'pretrained': False},
        'attention_config': {'attention_type': 'cbam'},
        'yolo_config': {'nc': 6}
    }
    
    model = ModelFactory.create_model('tinivit_yolo', config)
    print(f"TinyViT-YOLO created: {type(model).__name__}")
    
    # Test with different configurations
    for config_name in ['tinivit_yolo_small', 'tinivit_yolo_medium']:
        model = create_model_from_config(config_name, {'training_stage': 'foundation'})
        print(f"Model '{config_name}' created successfully")
    
    print("\nModel factory tests passed!")

if __name__ == "__main__":
    test_model_factory()
    compare_model_sizes()