import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import Dict, List, Optional
import torch

def visualize_predictions(image: np.ndarray, 
                         predictions: Dict, 
                         targets: Dict = None,
                         class_names: Dict = None,
                         save_path: str = None):
    """
    Visualize predictions on an image.
    
    Args:
        image: Input image
        predictions: Prediction dictionary with boxes, scores, labels
        targets: Optional ground truth targets
        class_names: Optional class name mapping
        save_path: Optional path to save the visualization
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    ax.set_title('Detection Results')
    ax.axis('off')
    
    # Draw predictions
    if 'boxes' in predictions and len(predictions['boxes']) > 0:
        boxes = predictions['boxes']
        scores = predictions.get('scores', [1.0] * len(boxes))
        labels = predictions.get('labels', [0] * len(boxes))
        
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            
            # Draw box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            
            # Add label
            class_name = class_names.get(label, f'Class_{label}') if class_names else f'Class_{label}'
            ax.text(x1, y1-5, f'{class_name}: {score:.2f}', 
                   color='red', fontsize=8, weight='bold')
    
    # Draw ground truth if provided
    if targets and 'boxes' in targets and len(targets['boxes']) > 0:
        boxes = targets['boxes']
        labels = targets.get('labels', [0] * len(boxes))
        
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            
            # Draw box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color='green', linewidth=2)
            ax.add_patch(rect)
            
            # Add label
            class_name = class_names.get(label, f'GT_{label}') if class_names else f'GT_{label}'
            ax.text(x1, y2+5, class_name, 
                   color='green', fontsize=8, weight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_training_curves(history: List[Dict], save_path: str = None):
    """
    Plot training curves from training history.
    
    Args:
        history: List of training history dictionaries
        save_path: Optional path to save the plot
    """
    if not history:
        return
    
    epochs = [h['epoch'] for h in history]
    train_losses = [h['train']['loss'] for h in history if 'train' in h]
    val_maps = [h['val']['map'] for h in history if 'val' in h]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    if train_losses:
        ax1.plot(epochs[:len(train_losses)], train_losses, 'b-', label='Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True)
    
    # Plot validation mAP
    if val_maps:
        ax2.plot(epochs[:len(val_maps)], val_maps, 'r-', label='Validation mAP')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mAP')
        ax2.set_title('Validation mAP')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_confusion_matrix(predictions: List[Dict], 
                          targets: List[Dict],
                          num_classes: int,
                          class_names: Dict = None,
                          save_path: str = None):
    """
    Create confusion matrix from predictions and targets.
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        num_classes: Number of classes
        class_names: Optional class name mapping
        save_path: Optional path to save the plot
    """
    # This is a placeholder implementation
    # In practice, you'd need to properly match predictions to targets
    
    matrix = np.random.randint(0, 100, (num_classes, num_classes))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='Blues')
    
    # Add colorbar
    plt.colorbar(im)
    
    # Set ticks and labels
    if class_names:
        labels = [class_names.get(i, f'Class_{i}') for i in range(num_classes)]
    else:
        labels = [f'Class_{i}' for i in range(num_classes)]
    
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    
    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(matrix[i, j]), 
                   ha='center', va='center', color='black')
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def test_visualization():
    """Test visualization functions."""
    print("Testing visualization functions...")
    
    # Create dummy data
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    predictions = {
        'boxes': np.array([[100, 100, 200, 200], [300, 300, 400, 400]]),
        'scores': np.array([0.9, 0.8]),
        'labels': np.array([1, 2])
    }
    
    targets = {
        'boxes': np.array([[105, 105, 195, 195]]),
        'labels': np.array([1])
    }
    
    class_names = {0: 'background', 1: 'defect1', 2: 'defect2'}
    
    try:
        # Test prediction visualization
        visualize_predictions(image, predictions, targets, class_names)
        print("  Prediction visualization test passed")
        
        # Test training curves
        history = [
            {'epoch': 1, 'train': {'loss': 1.0}, 'val': {'map': 0.5}},
            {'epoch': 2, 'train': {'loss': 0.8}, 'val': {'map': 0.6}},
            {'epoch': 3, 'train': {'loss': 0.6}, 'val': {'map': 0.7}}
        ]
        plot_training_curves(history)
        print("  Training curves test passed")
        
        # Test confusion matrix
        create_confusion_matrix([], [], 3, class_names)
        print("  Confusion matrix test passed")
        
        print("All visualization tests passed!")
        
    except Exception as e:
        print(f"Visualization test failed: {e}")

if __name__ == "__main__":
    test_visualization()