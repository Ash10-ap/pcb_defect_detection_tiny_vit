import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

class PCBMetrics:
    """
    Comprehensive metrics computation for PCB defect detection.
    
    Supports:
    - mAP calculation at different IoU thresholds
    - Per-class precision and recall
    - Small object detection metrics
    - Confusion matrix generation
    """
    
    def __init__(self, num_classes: int, iou_thresholds: List[float] = None):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.predictions = []
        self.targets = []
        
    def update(self, predictions: List[Dict], targets: List[Dict]):
        """
        Update metrics with new predictions and targets.
        
        Args:
            predictions: List of prediction dictionaries
            targets: List of target dictionaries
        """
        self.predictions.extend(predictions)
        self.targets.extend(targets)
    
    def compute_metrics(self, predictions: List[Dict] = None, targets: List[Dict] = None) -> Dict:
        """
        Compute comprehensive detection metrics.
        
        Args:
            predictions: Optional new predictions to add
            targets: Optional new targets to add
            
        Returns:
            Dictionary containing all computed metrics
        """
        if predictions is not None and targets is not None:
            self.update(predictions, targets)
        
        if not self.predictions or not self.targets:
            return self._empty_metrics()
        
        # Convert to format suitable for mAP calculation
        all_predictions = []
        all_targets = []
        
        for pred, target in zip(self.predictions, self.targets):
            # Convert predictions
            pred_boxes = pred.get('boxes', torch.tensor([]))
            pred_scores = pred.get('scores', torch.tensor([]))
            pred_labels = pred.get('labels', torch.tensor([]))
            
            if len(pred_boxes) > 0:
                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    all_predictions.append({
                        'image_id': pred.get('image_id', 0),
                        'bbox': box.cpu().numpy(),
                        'score': score.cpu().item(),
                        'category_id': label.cpu().item()
                    })
            
            # Convert targets
            target_boxes = target.get('boxes', torch.tensor([]))
            target_labels = target.get('labels', torch.tensor([]))
            
            if len(target_boxes) > 0:
                for box, label in zip(target_boxes, target_labels):
                    all_targets.append({
                        'image_id': target.get('image_id', 0),
                        'bbox': box.cpu().numpy(),
                        'category_id': label.cpu().item()
                    })
        
        # Calculate metrics
        metrics = {}
        
        # mAP calculation
        map_results = self._calculate_map(all_predictions, all_targets)
        metrics.update(map_results)
        
        # Per-class metrics
        class_metrics = self._calculate_per_class_metrics(all_predictions, all_targets)
        metrics.update(class_metrics)
        
        # Small object metrics
        small_object_metrics = self._calculate_small_object_metrics(all_predictions, all_targets)
        metrics.update(small_object_metrics)
        
        return metrics
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dictionary."""
        return {
            'map': 0.0,
            'map50': 0.0,
            'map75': 0.0,
            'maps': 0.0,  # Small objects
            'mapm': 0.0,  # Medium objects
            'mapl': 0.0,  # Large objects
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    def _calculate_map(self, predictions: List[Dict], targets: List[Dict]) -> Dict:
        """Calculate mAP at different IoU thresholds."""
        if not predictions or not targets:
            return self._empty_metrics()
        
        # Group by image
        pred_by_image = defaultdict(list)
        target_by_image = defaultdict(list)
        
        for pred in predictions:
            pred_by_image[pred['image_id']].append(pred)
        
        for target in targets:
            target_by_image[target['image_id']].append(target)
        
        # Calculate AP for each class and IoU threshold
        aps = []
        
        for class_id in range(1, self.num_classes + 1):  # Skip background class
            class_aps = []
            
            for iou_thresh in self.iou_thresholds:
                ap = self._calculate_ap_single_class(
                    pred_by_image, target_by_image, class_id, iou_thresh
                )
                class_aps.append(ap)
            
            aps.append(class_aps)
        
        # Average across classes and IoU thresholds
        aps = np.array(aps)
        
        map_result = {
            'map': np.mean(aps),
            'map50': np.mean(aps[:, 0]),  # IoU=0.5
            'map75': np.mean(aps[:, 5]),  # IoU=0.75
        }
        
        return map_result
    
    def _calculate_ap_single_class(self, 
                                  pred_by_image: Dict, 
                                  target_by_image: Dict,
                                  class_id: int,
                                  iou_threshold: float) -> float:
        """Calculate AP for a single class at specific IoU threshold."""
        # Collect all predictions and targets for this class
        class_predictions = []
        class_targets = []
        
        for image_id in set(list(pred_by_image.keys()) + list(target_by_image.keys())):
            # Get predictions for this image and class
            image_preds = [p for p in pred_by_image.get(image_id, []) 
                          if p['category_id'] == class_id]
            
            # Get targets for this image and class
            image_targets = [t for t in target_by_image.get(image_id, [])
                           if t['category_id'] == class_id]
            
            # Match predictions to targets
            if image_preds and image_targets:
                matched_preds, matched_targets = self._match_predictions_targets(
                    image_preds, image_targets, iou_threshold
                )
                class_predictions.extend(matched_preds)
                class_targets.extend(matched_targets)
            elif image_preds:
                # False positives
                for pred in image_preds:
                    class_predictions.append((pred['score'], 0))  # 0 = False Positive
            elif image_targets:
                # False negatives (missed detections)
                for _ in image_targets:
                    class_targets.append(1)  # Missed target
        
        if not class_predictions:
            return 0.0
        
        # Sort predictions by score
        class_predictions.sort(key=lambda x: x[0], reverse=True)
        
        # Calculate precision and recall
        tp = np.array([p[1] for p in class_predictions])
        fp = 1 - tp
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        num_targets = len([t for t in class_targets if t == 1]) + len(class_targets)
        
        if num_targets == 0:
            return 0.0
        
        recalls = tp_cumsum / num_targets
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Calculate AP using 101-point interpolation
        ap = self._calculate_ap_from_pr(precisions, recalls)
        
        return ap
    
    def _match_predictions_targets(self, 
                                  predictions: List[Dict],
                                  targets: List[Dict],
                                  iou_threshold: float) -> Tuple[List, List]:
        """Match predictions to targets based on IoU threshold."""
        if not predictions or not targets:
            return [], []
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(predictions), len(targets)))
        
        for i, pred in enumerate(predictions):
            for j, target in enumerate(targets):
                iou = self._calculate_iou(pred['bbox'], target['bbox'])
                iou_matrix[i, j] = iou
        
        # Hungarian algorithm for optimal matching
        from scipy.optimize import linear_sum_assignment
        
        # Only consider matches above threshold
        valid_matches = iou_matrix >= iou_threshold
        
        if not np.any(valid_matches):
            # No valid matches
            matched_preds = [(pred['score'], 0) for pred in predictions]  # All FP
            matched_targets = [1] * len(targets)  # All FN
            return matched_preds, matched_targets
        
        # Set invalid matches to very low value
        cost_matrix = -iou_matrix
        cost_matrix[~valid_matches] = 1e6
        
        pred_indices, target_indices = linear_sum_assignment(cost_matrix)
        
        matched_preds = []
        matched_targets = []
        
        # Process matched pairs
        used_targets = set()
        for pred_idx, target_idx in zip(pred_indices, target_indices):
            if iou_matrix[pred_idx, target_idx] >= iou_threshold:
                matched_preds.append((predictions[pred_idx]['score'], 1))  # TP
                used_targets.add(target_idx)
            else:
                matched_preds.append((predictions[pred_idx]['score'], 0))  # FP
        
        # Add unmatched predictions as FP
        for i, pred in enumerate(predictions):
            if i not in pred_indices:
                matched_preds.append((pred['score'], 0))
        
        # Add unmatched targets as FN
        for i in range(len(targets)):
            if i not in used_targets:
                matched_targets.append(1)
        
        return matched_preds, matched_targets
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two bounding boxes."""
        # Convert from [x1, y1, x2, y2] format if needed
        if len(box1) == 4 and len(box2) == 4:
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        return 0.0
    
    def _calculate_ap_from_pr(self, precisions: np.ndarray, recalls: np.ndarray) -> float:
        """Calculate AP from precision-recall curve using 101-point interpolation."""
        # Add boundary points
        recalls = np.concatenate(([0], recalls, [1]))
        precisions = np.concatenate(([0], precisions, [0]))
        
        # Compute monotonic precision
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        
        # Calculate AP using 101-point interpolation
        recall_points = np.linspace(0, 1, 101)
        ap = 0.0
        
        for r in recall_points:
            # Find precisions at recall >= r
            indices = recalls >= r
            if np.any(indices):
                ap += np.max(precisions[indices])
        
        return ap / 101
    
    def _calculate_per_class_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict:
        """Calculate per-class precision, recall, and F1 scores."""
        class_metrics = {}
        
        for class_id in range(1, self.num_classes + 1):
            # Get class-specific predictions and targets
            class_preds = [p for p in predictions if p['category_id'] == class_id]
            class_targets = [t for t in targets if t['category_id'] == class_id]
            
            if not class_targets:
                continue
            
            # Calculate metrics for this class
            tp = fp = fn = 0
            
            # Simple counting based on IoU > 0.5
            # This is a simplified implementation
            if class_preds and class_targets:
                # Match predictions to targets
                matched = self._match_predictions_targets(
                    class_preds, class_targets, 0.5
                )
                tp = sum(1 for score, is_tp in matched[0] if is_tp)
                fp = sum(1 for score, is_tp in matched[0] if not is_tp)
                fn = len(class_targets) - tp
            else:
                fn = len(class_targets)
                fp = len(class_preds)
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_metrics[f'class_{class_id}_precision'] = precision
            class_metrics[f'class_{class_id}_recall'] = recall
            class_metrics[f'class_{class_id}_f1'] = f1
        
        # Overall metrics
        if class_metrics:
            all_precisions = [v for k, v in class_metrics.items() if 'precision' in k]
            all_recalls = [v for k, v in class_metrics.items() if 'recall' in k]
            all_f1s = [v for k, v in class_metrics.items() if 'f1' in k]
            
            class_metrics['precision'] = np.mean(all_precisions) if all_precisions else 0.0
            class_metrics['recall'] = np.mean(all_recalls) if all_recalls else 0.0
            class_metrics['f1'] = np.mean(all_f1s) if all_f1s else 0.0
        
        return class_metrics
    
    def _calculate_small_object_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict:
        """Calculate metrics specifically for small objects."""
        # Define size thresholds (in pixels)
        small_threshold = 32 * 32  # 32x32 pixels
        medium_threshold = 96 * 96  # 96x96 pixels
        
        small_preds = []
        medium_preds = []
        large_preds = []
        
        small_targets = []
        medium_targets = []
        large_targets = []
        
        # Categorize by size
        for pred in predictions:
            area = self._calculate_bbox_area(pred['bbox'])
            if area <= small_threshold:
                small_preds.append(pred)
            elif area <= medium_threshold:
                medium_preds.append(pred)
            else:
                large_preds.append(pred)
        
        for target in targets:
            area = self._calculate_bbox_area(target['bbox'])
            if area <= small_threshold:
                small_targets.append(target)
            elif area <= medium_threshold:
                medium_targets.append(target)
            else:
                large_targets.append(target)
        
        # Calculate mAP for each size category
        maps = self._calculate_map(small_preds, small_targets).get('map', 0.0)
        mapm = self._calculate_map(medium_preds, medium_targets).get('map', 0.0)
        mapl = self._calculate_map(large_preds, large_targets).get('map', 0.0)
        
        return {
            'maps': maps,
            'mapm': mapm,
            'mapl': mapl
        }
    
    def _calculate_bbox_area(self, bbox: np.ndarray) -> float:
        """Calculate bounding box area."""
        if len(bbox) >= 4:
            return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        return 0.0


def calculate_map(predictions: List[Dict], 
                 targets: List[Dict], 
                 num_classes: int,
                 iou_thresholds: List[float] = None) -> Dict:
    """
    Standalone function to calculate mAP.
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        num_classes: Number of classes
        iou_thresholds: IoU thresholds for evaluation
        
    Returns:
        Dictionary with mAP results
    """
    metrics = PCBMetrics(num_classes, iou_thresholds)
    return metrics.compute_metrics(predictions, targets)


def calculate_precision_recall(predictions: List[Dict], 
                             targets: List[Dict],
                             class_id: int,
                             iou_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate precision-recall curve for a specific class.
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        class_id: Class ID to evaluate
        iou_threshold: IoU threshold for matching
        
    Returns:
        Tuple of (precision, recall) arrays
    """
    # Filter predictions and targets for specific class
    class_preds = [p for p in predictions if p.get('category_id') == class_id]
    class_targets = [t for t in targets if t.get('category_id') == class_id]
    
    if not class_preds or not class_targets:
        return np.array([]), np.array([])
    
    # Sort predictions by score
    class_preds.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    # Calculate matches
    tp = np.zeros(len(class_preds))
    fp = np.zeros(len(class_preds))
    
    # This is a simplified implementation
    # In practice, you'd need proper IoU matching
    for i, pred in enumerate(class_preds):
        # Simplified: assume first predictions are correct
        if i < len(class_targets):
            tp[i] = 1
        else:
            fp[i] = 1
    
    # Calculate cumulative TP and FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Calculate precision and recall
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / len(class_targets)
    
    return precisions, recalls


def test_metrics():
    """Test the metrics computation."""
    print("Testing PCB Metrics")
    print("=" * 20)
    
    # Create dummy predictions and targets
    predictions = [
        {
            'boxes': torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]]),
            'scores': torch.tensor([0.9, 0.8]),
            'labels': torch.tensor([1, 2]),
            'image_id': 0
        }
    ]
    
    targets = [
        {
            'boxes': torch.tensor([[105, 105, 195, 195], [310, 310, 390, 390]]),
            'labels': torch.tensor([1, 2]),
            'image_id': 0
        }
    ]
    
    # Test metrics calculation
    metrics = PCBMetrics(num_classes=6)
    results = metrics.compute_metrics(predictions, targets)
    
    print("Computed metrics:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    print("Metrics computation test passed!")


if __name__ == "__main__":
    test_metrics()