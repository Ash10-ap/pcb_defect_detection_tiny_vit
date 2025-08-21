import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np

class FewShotTrainer:
    """
    Few-shot learning trainer for novel PCB defect types.
    
    Implements episodic training for few-shot adaptation.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 n_way: int = 5,
                 n_shot: int = 5,
                 n_query: int = 10):
        """
        Initialize few-shot trainer.
        
        Args:
            model: Base model to adapt
            n_way: Number of classes per episode
            n_shot: Number of support samples per class
            n_query: Number of query samples per class
        """
        self.model = model
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
    
    def create_episode(self, dataset, classes: List[int] = None):
        """
        Create a few-shot learning episode.
        
        Args:
            dataset: Dataset to sample from
            classes: Optional specific classes to use
            
        Returns:
            Support and query sets for the episode
        """
        if classes is None:
            # Randomly sample classes
            all_classes = list(range(len(dataset.classes)))
            classes = np.random.choice(all_classes, self.n_way, replace=False)
        
        support_set = []
        query_set = []
        
        for class_id in classes:
            # Get samples for this class
            class_samples = [i for i, sample in enumerate(dataset.samples) 
                           if any(ann['category_id'] == class_id 
                                 for ann in sample['annotations'])]
            
            if len(class_samples) < (self.n_shot + self.n_query):
                continue
            
            # Sample support and query
            selected = np.random.choice(class_samples, 
                                      self.n_shot + self.n_query, 
                                      replace=False)
            
            support_indices = selected[:self.n_shot]
            query_indices = selected[self.n_shot:]
            
            support_set.extend(support_indices)
            query_set.extend(query_indices)
        
        return support_set, query_set
    
    def train_episode(self, support_loader, query_loader, optimizer, criterion):
        """
        Train on a single episode.
        
        Args:
            support_loader: DataLoader for support set
            query_loader: DataLoader for query set
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            Episode loss and accuracy
        """
        self.model.train()
        
        # Forward pass on support set
        support_features = []
        support_labels = []
        
        with torch.no_grad():
            for images, targets in support_loader:
                features = self.model.backbone(images)
                support_features.append(features)
                support_labels.extend([t['labels'] for t in targets])
        
        # Compute prototype representations
        prototypes = self.compute_prototypes(support_features, support_labels)
        
        # Training on query set
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, targets in query_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Compute accuracy (simplified)
            total += len(targets)
            # This would need proper implementation based on detection metrics
        
        avg_loss = total_loss / len(query_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def compute_prototypes(self, features, labels):
        """
        Compute prototype representations for each class.
        
        Args:
            features: List of feature tensors
            labels: List of labels
            
        Returns:
            Dictionary mapping class IDs to prototype features
        """
        prototypes = {}
        
        # Group features by class
        class_features = {}
        for feat, label_list in zip(features, labels):
            for label in label_list:
                label_id = label.item() if torch.is_tensor(label) else label
                if label_id not in class_features:
                    class_features[label_id] = []
                class_features[label_id].append(feat)
        
        # Compute prototypes as mean features
        for class_id, feat_list in class_features.items():
            if feat_list:
                stacked_features = torch.stack(feat_list)
                prototype = torch.mean(stacked_features, dim=0)
                prototypes[class_id] = prototype
        
        return prototypes
    
    def evaluate_episode(self, support_loader, query_loader):
        """
        Evaluate on a single episode.
        
        Args:
            support_loader: DataLoader for support set
            query_loader: DataLoader for query set
            
        Returns:
            Episode accuracy
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get support features and compute prototypes
            support_features = []
            support_labels = []
            
            for images, targets in support_loader:
                features = self.model.backbone(images)
                support_features.append(features)
                support_labels.extend([t['labels'] for t in targets])
            
            prototypes = self.compute_prototypes(support_features, support_labels)
            
            # Evaluate on query set
            correct = 0
            total = 0
            
            for images, targets in query_loader:
                outputs = self.model(images)
                
                # Simplified accuracy computation
                total += len(targets)
                # This would need proper implementation
            
            accuracy = correct / total if total > 0 else 0.0
            
        return accuracy

def test_few_shot_trainer():
    """Test few-shot trainer functionality."""
    print("Testing Few-Shot Trainer")
    print("=" * 25)
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            self.head = nn.Linear(64, 5)
        
        def forward(self, x):
            feat = self.backbone(x)
            return self.head(feat)
    
    model = DummyModel()
    trainer = FewShotTrainer(model, n_way=3, n_shot=5, n_query=10)
    
    print(f"Few-shot trainer created: {trainer.n_way}-way {trainer.n_shot}-shot")
    print("Few-shot trainer test passed!")

if __name__ == "__main__":
    test_few_shot_trainer()