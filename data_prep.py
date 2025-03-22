import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
import copy
import random

class GTSRBWrapper(Dataset):
    """Custom wrapper for GTSRB to extract targets easily"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.targets = self._extract_targets()
        
    def _extract_targets(self):
        targets = []
        for i in range(len(self.dataset)):
            _, target = self.dataset[i]
            targets.append(target)
        return targets
    
    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)


def create_task_splits(dataset, num_tasks=5):
    """ Create task splits for GTSRB """
    num_classes = 43  # GTSRB has 43 classes
    classes_per_task = num_classes // num_tasks
    
    # Create class to sample mapping
    class_to_idx = {}
    for idx, target in enumerate(dataset.targets):
        if target not in class_to_idx:
            class_to_idx[target] = []
        class_to_idx[target].append(idx)
    
    task_datasets = []
    for task_id in range(num_tasks):
        start_class = task_id * classes_per_task
        end_class = (task_id + 1) * classes_per_task if task_id < num_tasks - 1 else num_classes
        
        indices = []
        for class_id in range(start_class, end_class):
            indices.extend(class_to_idx.get(class_id, []))
        
        task_datasets.append(Subset(dataset, indices))
    
    return task_datasets
