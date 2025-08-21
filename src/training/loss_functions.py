import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Optional imports for specific YOLO components
try:
    from ultralytics.nn.modules import Detect
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

class SIoULoss(nn.Module):
    """
    SIoU (SCYLLA-IoU) Loss for bounding box regression.
    
    More accurate than standard IoU for small objects and overlapping boxes.
    Particularly useful for PCB defect detection where defects can be small
    and closely packed.
    
    Reference: "SIoU Loss: More Powerful Learning for Bounding Box Regression"
    """
    
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: (N, 4) tensor in format [x1, y1, x2, y2]
            target_boxes: (N, 4) tensor in format [x1, y1, x2, y2]
        
        Returns:
            SIoU loss value
        """
        # Calculate intersection coordinates
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        # Calculate intersection area
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                    torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Calculate individual box areas
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * \
                   (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * \
                     (target_boxes[:, 3] - target_boxes[:, 1])
        
        # Calculate union area
        union_area = pred_area + target_area - inter_area + self.eps
        
        # Basic IoU
        iou = inter_area / union_area
        
        # Calculate centers
        pred_center_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        pred_center_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        target_center_x = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
        target_center_y = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
        
        # Distance between centers
        center_distance = (pred_center_x - target_center_x) ** 2 + \
                         (pred_center_y - target_center_y) ** 2
        
        # Diagonal of smallest enclosing box
        enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        
        enclose_diagonal = (enclose_x2 - enclose_x1) ** 2 + \
                          (enclose_y2 - enclose_y1) ** 2 + self.eps
        
        # Angle cost
        sigma = torch.pow(center_distance / enclose_diagonal, 0.5)
        sin_alpha = torch.abs(pred_center_x - target_center_x) / \
                   (torch.sqrt(center_distance) + self.eps)
        sin_beta = torch.abs(pred_center_y - target_center_y) / \
                  (torch.sqrt(center_distance) + self.eps)
        
        threshold = pow(2, 0.5) / 2
        sin_alpha = torch.where(sin_alpha > threshold, 
                               torch.cos(torch.arcsin(sin_alpha) - math.pi / 4),
                               sin_alpha)
        
        angle_cost = 2 * torch.pow(torch.sin(torch.arcsin(sin_alpha) - math.pi / 4), 2)
        
        # Shape cost
        pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
        pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
        target_w = target_boxes[:, 2] - target_boxes[:, 0]
        target_h = target_boxes[:, 3] - target_boxes[:, 1]
        
        theta = 4 / (math.pi ** 2) * torch.pow(
            torch.arctan(target_w / (target_h + self.eps)) - 
            torch.arctan(pred_w / (pred_h + self.eps)), 2
        )
        
        shape_cost = 1 - torch.exp(-1 * theta)
        
        # Distance cost
        rho = center_distance / enclose_diagonal
        distance_cost = 2 - torch.exp(-1 * sigma) * (2 - rho)
        
        # Final SIoU
        siou = iou - (distance_cost + shape_cost) / 2
        
        return 1 - siou.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in object detection.
    
    Particularly useful for PCB defect detection where some defect types
    are much rarer than others.
    
    Reference: "Focal Loss for Dense Object Detection" (RetinaNet paper)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) tensor of raw predictions
            targets: (N,) tensor of class labels
        
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Apply focal term
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for YOLOv8-style object detection.
    
    Combines:
    - SIoU loss for bounding box regression
    - Focal loss for classification
    - Binary cross-entropy for objectness
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.bbox_weight = config.get('bbox_weight', 7.5)
        self.cls_weight = config.get('cls_weight', 0.5)
        self.obj_weight = config.get('obj_weight', 1.0)
        
        self.siou_loss = SIoULoss()
        self.focal_loss = FocalLoss(
            alpha=config.get('focal_alpha', 0.25),
            gamma=config.get('focal_gamma', 1.5)
        )
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: List of prediction tensors from model
            targets: List of target dictionaries
        
        Returns:
            Combined loss value and loss components
        """
        total_bbox_loss = 0
        total_cls_loss = 0
        total_obj_loss = 0
        
        num_targets = 0
        
        for pred, target in zip(predictions, targets):
            if len(target['boxes']) == 0:
                continue
                
            # Extract prediction components
            pred_boxes = pred[..., :4]
            pred_obj = pred[..., 4]
            pred_cls = pred[..., 5:]
            
            # Reshape predictions
            batch_size, num_anchors, grid_h, grid_w = pred.shape[:4]
            pred_boxes = pred_boxes.view(batch_size, -1, 4)
            pred_obj = pred_obj.view(batch_size, -1)
            pred_cls = pred_cls.view(batch_size, -1, pred_cls.shape[-1])
            
            # Get positive samples (simplified)
            target_boxes = target['boxes']
            target_labels = target['labels']
            
            if len(target_boxes) > 0:
                # Bounding box loss (using SIoU)
                bbox_loss = self.siou_loss(pred_boxes[0][:len(target_boxes)], 
                                         target_boxes)
                total_bbox_loss += bbox_loss
                
                # Classification loss (using Focal Loss)
                cls_loss = self.focal_loss(pred_cls[0][:len(target_labels)], 
                                         target_labels)
                total_cls_loss += cls_loss
                
                # Objectness loss (simplified)
                obj_targets = torch.zeros_like(pred_obj[0])
                obj_targets[:len(target_boxes)] = 1.0
                obj_loss = self.bce_loss(pred_obj[0], obj_targets)
                total_obj_loss += obj_loss
                
                num_targets += len(target_boxes)
        
        # Normalize by number of targets
        if num_targets > 0:
            total_bbox_loss /= num_targets
            total_cls_loss /= num_targets
            total_obj_loss /= num_targets
        
        # Combine losses
        total_loss = (self.bbox_weight * total_bbox_loss + 
                     self.cls_weight * total_cls_loss + 
                     self.obj_weight * total_obj_loss)
        
        return total_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing cross-entropy loss.
    
    Helps prevent overfitting and improves generalization,
    especially useful for small datasets.
    """
    
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        """
        Args:
            pred: (N, C) tensor of predictions
            target: (N,) tensor of targets
        """
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=1)
        
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class AdaptiveLoss(nn.Module):
    """
    Adaptive loss that adjusts weights based on training progress.
    
    Gradually shifts focus from bbox regression to classification
    as training progresses.
    """
    
    def __init__(self, config):
        super().__init__()
        self.combined_loss = CombinedLoss(config)
        self.initial_bbox_weight = config.get('bbox_weight', 7.5)
        self.initial_cls_weight = config.get('cls_weight', 0.5)
        self.current_epoch = 0
        self.total_epochs = config.get('total_epochs', 100)
        
    def update_epoch(self, epoch):
        """Update current epoch for adaptive weighting."""
        self.current_epoch = epoch
        
        # Gradually reduce bbox weight and increase cls weight
        progress = epoch / self.total_epochs
        bbox_decay = 0.5 * progress
        cls_increase = 0.5 * progress
        
        self.combined_loss.bbox_weight = self.initial_bbox_weight * (1 - bbox_decay)
        self.combined_loss.cls_weight = self.initial_cls_weight * (1 + cls_increase)
        
    def forward(self, predictions, targets):
        return self.combined_loss(predictions, targets)


def test_loss_functions():
    """Test the loss functions."""
    print("Testing Loss Functions")
    print("=" * 25)
    
    # Test SIoU Loss
    pred_boxes = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],
        [5.0, 5.0, 15.0, 15.0]
    ])
    target_boxes = torch.tensor([
        [1.0, 1.0, 11.0, 11.0],
        [4.0, 4.0, 14.0, 14.0]
    ])
    
    siou_loss = SIoULoss()
    loss_value = siou_loss(pred_boxes, target_boxes)
    print(f"SIoU Loss: {loss_value.item():.4f}")
    
    # Test Focal Loss
    inputs = torch.randn(10, 5)
    targets = torch.randint(0, 5, (10,))
    
    focal_loss = FocalLoss()
    loss_value = focal_loss(inputs, targets)
    print(f"Focal Loss: {loss_value.item():.4f}")
    
    # Test Combined Loss
    config = {'bbox_weight': 1.0, 'cls_weight': 1.0, 'obj_weight': 1.0}
    combined_loss = CombinedLoss(config)
    
    # Mock predictions and targets
    predictions = [torch.randn(1, 25200, 11)]  # Batch=1, Anchors=25200, Features=11
    targets = [{'boxes': torch.tensor([[0, 0, 10, 10]]), 'labels': torch.tensor([1])}]
    
    try:
        loss_value = combined_loss(predictions, targets)
        print(f"Combined Loss: {loss_value.item():.4f}")
    except Exception as e:
        print(f"Combined Loss: Error - {e}")
    
    print("Loss function tests completed!")


if __name__ == "__main__":
    test_loss_functions()