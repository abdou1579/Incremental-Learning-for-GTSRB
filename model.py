import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy
import random


# Model architecture
"""Mode Architecture, for features extration """
class GTSRB_extractor(nn.Module):
    def __init__(self, num_classes=43):
        super(GTSRB_extractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(self.bn1(self.relu(self.conv1(x))))
        x = self.pool(self.bn2(self.relu(self.conv2(x))))
        x = self.pool(self.bn3(self.relu(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
    def features(self, x):
        x = self.pool(self.bn1(self.relu(self.conv1(x))))
        x = self.pool(self.bn2(self.relu(self.conv2(x))))
        x = self.pool(self.bn3(self.relu(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = self.relu(self.fc1(x))
        return x

class MemoryBuffer:
    """Memory Buffer for rehearsal"""
    #See https://github.com/zhoudw-zdw/CIL_Survey/
    def __init__(self, buffer_size=200):
        self.buffer_size = buffer_size
        self.images = []
        self.labels = []
        
    def update(self, images, labels, per_class=20):
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            indices = torch.where(labels == label)[0]
            selected = indices[:min(per_class, len(indices))]
            
            self.images.extend(images[selected].cpu())
            self.labels.extend(labels[selected].cpu())
            
        # Keep only buffer_size samples
        if len(self.images) > self.buffer_size:
            indices = random.sample(range(len(self.images)), self.buffer_size)
            self.images = [self.images[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
    
    def get_data(self):
        if len(self.images) == 0:
            return None, None
        return torch.stack(self.images), torch.tensor(self.labels)





def distillation_loss(outputs, teacher_outputs, T=2.0):
    """Knowledge Distillation Loss"""
    soft_targets = nn.functional.softmax(teacher_outputs / T, dim=1)
    soft_prob = nn.functional.log_softmax(outputs / T, dim=1)
    return nn.KLDivLoss(reduction='batchmean')(soft_prob, soft_targets) * (T * T)
