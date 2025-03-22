import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader

def plot_accuracy_matrix(accuracy_matrix, title="Task Accuracy Matrix"):
    """
    Plot a heatmap of the accuracy matrix showing performance on each task
    after learning each subsequent task.
    
    Args:
        accuracy_matrix: NumPy array of shape (num_tasks, num_tasks)
        title: Title for the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Create a mask for the upper triangle (future tasks)
    mask = np.triu(np.ones_like(accuracy_matrix, dtype=bool), k=1)
    
    # Define task names for better labeling
    task_labels = [f"Task {i+1}" for i in range(accuracy_matrix.shape[0])]
    
    # Create heatmap
    sns.heatmap(accuracy_matrix, annot=True, fmt=".1f", cmap="YlGnBu",
                xticklabels=task_labels, yticklabels=task_labels,
                mask=mask, vmin=0, vmax=100)
    
    plt.xlabel("Evaluated on Task")
    plt.ylabel("After Learning Task")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("accuracy_matrix.png")
    plt.show()

def plot_accuracy_by_task(accuracy_matrix):
    """
    Plot line charts showing accuracy for each task over time.
    
    Args:
        accuracy_matrix: NumPy array of shape (num_tasks, num_tasks)
    """
    plt.figure(figsize=(12, 6))
    num_tasks = accuracy_matrix.shape[0]
    
    # Plot accuracy for each task
    for task_id in range(num_tasks):
        # Get accuracy values for this task (column in the matrix)
        task_accuracies = [accuracy_matrix[i, task_id] if i >= task_id else None 
                          for i in range(num_tasks)]
        
        # Plot line
        plt.plot(range(task_id + 1, num_tasks + 1), task_accuracies[task_id:], 
                 marker='o', label=f"Task {task_id+1}")
    
    plt.xlabel("After Learning Task")
    plt.ylabel("Accuracy (%)")
    plt.title("Task Performance Throughout Incremental Learning")
    plt.xticks(range(1, num_tasks + 1))
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("accuracy_by_task.png")
    plt.show()

def plot_forgetting(accuracy_matrix):
    """
    Plot the forgetting (difference between max and final accuracy) for each task.
    
    Args:
        accuracy_matrix: NumPy array of shape (num_tasks, num_tasks)
    """
    num_tasks = accuracy_matrix.shape[0]
    forgetting = np.zeros(num_tasks)
    
    for task_id in range(num_tasks):
        # Calculate max accuracy for this task and the final accuracy
        if task_id < num_tasks - 1:
            max_acc = np.max(accuracy_matrix[:task_id+1, task_id])
        else:
            max_acc = accuracy_matrix[task_id, task_id]
        
        final_acc = accuracy_matrix[num_tasks-1, task_id]
        forgetting[task_id] = max_acc - final_acc
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(1, num_tasks + 1), forgetting, color='skyblue')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    plt.xlabel("Task")
    plt.ylabel("Forgetting (%)")
    plt.title("Catastrophic Forgetting by Task")
    plt.xticks(range(1, num_tasks + 1))
    plt.ylim(0, max(forgetting) * 1.2 + 5)  # Add some padding
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("forgetting_by_task.png")
    plt.show()

def plot_all_metrics(accuracy_matrix):
    """
    Create all plots at once and display summary metrics.
    
    Args:
        accuracy_matrix: NumPy array of shape (num_tasks, num_tasks)
    """
    # Plot accuracy matrix heatmap
    plot_accuracy_matrix(accuracy_matrix)
    
    # Plot accuracy by task
    plot_accuracy_by_task(accuracy_matrix)
    
    # Plot forgetting
    plot_forgetting(accuracy_matrix)
    
    # Calculate and print summary metrics
    num_tasks = accuracy_matrix.shape[0]
    
    # Average accuracy after learning all tasks
    final_accuracy = np.mean(accuracy_matrix[num_tasks-1, :])
    
    # Average forgetting
    forgetting = np.zeros(num_tasks)
    for task_id in range(num_tasks):
        if task_id < num_tasks - 1:
            max_acc = np.max(accuracy_matrix[:task_id+1, task_id])
        else:
            max_acc = accuracy_matrix[task_id, task_id]
        forgetting[task_id] = max_acc - accuracy_matrix[num_tasks-1, task_id]
    
    avg_forgetting = np.mean(forgetting)
    
    print(f"Summary Metrics:")
    print(f"Average Final Accuracy: {final_accuracy:.2f}%")
    print(f"Average Forgetting: {avg_forgetting:.2f}%")
    print(f"Forgetting by Task: {', '.join([f'{f:.2f}%' for f in forgetting])}")

from torchvision import transforms
from torch.utils.data import DataLoader

def plot_sample_images(dataset, num_images=16, classes_per_row=4, title="Sample Images from GTSRB"):
    """
    Plot a grid of sample images from the GTSRB dataset.
    
    Args:
        dataset: GTSRB dataset
        num_images: Number of images to plot
        classes_per_row: Number of images per row
        title: Title for the plot
    """
    # Set up the figure
    fig, axes = plt.subplots(num_images // classes_per_row, classes_per_row, 
                             figsize=(15, 15 * num_images // classes_per_row // classes_per_row))
    
    # Flatten axes array for easier indexing
    axes = axes.flatten()
    
    # Generate random indices
    indices = random.sample(range(len(dataset)), num_images)
    
    # Plot each image
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        
        # Convert tensor to numpy for visualization
        if isinstance(img, torch.Tensor):
            # Convert from [C,H,W] to [H,W,C] and denormalize if needed
            img = img.permute(1, 2, 0).numpy()
            
            # Denormalize if the images were normalized
            mean = np.array([0.3337, 0.3064, 0.3171])
            std = np.array([0.2672, 0.2564, 0.2629])
            img = img * std + mean
            
            # Clip values to valid range
            img = np.clip(img, 0, 1)
        
        # Plot the image
        axes[i].imshow(img)
        axes[i].set_title(f"Class {label}")
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig("gtsrb_samples.png")
    plt.show()

def plot_class_examples(dataset, num_classes=10, samples_per_class=5):
    """
    Plot multiple examples from each of several classes
    
    Args:
        dataset: GTSRB dataset
        num_classes: Number of different classes to show
        samples_per_class: Number of samples to show per class
    """
    # Extract targets from the dataset
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    else:
        # Extract targets manually if not available directly
        targets = []
        for i in range(len(dataset)):
            _, target = dataset[i]
            targets.append(target)
    
    # Find samples for each class
    class_to_idx = {}
    for idx, target in enumerate(targets):
        if target not in class_to_idx:
            class_to_idx[target] = []
        class_to_idx[target].append(idx)
    
    # Pick classes to display (use first num_classes if available)
    available_classes = sorted(list(class_to_idx.keys()))[:num_classes]
    
    # Set up the figure
    fig, axes = plt.subplots(len(available_classes), samples_per_class, 
                             figsize=(samples_per_class * 3, len(available_classes) * 3))
    
    # Plot examples from each class
    for i, class_idx in enumerate(available_classes):
        # Get indices for this class
        indices = random.sample(class_to_idx[class_idx], 
                               min(samples_per_class, len(class_to_idx[class_idx])))
        
        for j, idx in enumerate(indices):
            img, _ = dataset[idx]
            
            # Convert tensor to numpy for visualization
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()
                
                # Denormalize if the images were normalized
                mean = np.array([0.3337, 0.3064, 0.3171])
                std = np.array([0.2672, 0.2564, 0.2629])
                img = img * std + mean
                
                # Clip values to valid range
                img = np.clip(img, 0, 1)
            
            # Plot in the appropriate subplot
            if len(available_classes) == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]
                
            ax.imshow(img)
            
            # Only add class label to the first image in each row
            if j == 0:
                ax.set_ylabel(f"Class {class_idx}")
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.suptitle("Examples by Class from GTSRB", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig("gtsrb_class_examples.png")
    plt.show()

def plot_class_distribution(dataset, title="Class Distribution in GTSRB"):
    """
    Plot the distribution of classes in the dataset
    
    Args:
        dataset: GTSRB dataset
        title: Title for the plot
    """
    # Extract targets from the dataset
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    else:
        # Extract targets manually if not available directly
        targets = []
        for i in range(len(dataset)):
            _, target = dataset[i]
            targets.append(target)
    
    # Count occurrences of each class
    unique_classes = sorted(set(targets))
    class_counts = [targets.count(c) for c in unique_classes]
    
    plt.figure(figsize=(15, 6))
    bars = plt.bar(unique_classes, class_counts, alpha=0.8)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Class ID')
    plt.ylabel('Number of Samples')
    plt.title(title)
    plt.xticks(unique_classes)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    plt.show()

def plot_task_examples(train_tasks, num_samples=5):
    """
    Plot examples from each task to show how classes are distributed
    
    Args:
        train_tasks: List of task datasets
        num_samples: Number of samples to show per task
    """
    num_tasks = len(train_tasks)
    
    # Set up the figure
    fig, axes = plt.subplots(num_tasks, num_samples, figsize=(num_samples * 3, num_tasks * 3))
    
    for task_id, task_dataset in enumerate(train_tasks):
        # Get random indices
        indices = random.sample(range(len(task_dataset)), min(num_samples, len(task_dataset)))
        
        for j, idx in enumerate(indices):
            img, label = task_dataset[idx]
            
            # Convert tensor to numpy for visualization
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()
                
                # Denormalize if the images were normalized
                mean = np.array([0.3337, 0.3064, 0.3171])
                std = np.array([0.2672, 0.2564, 0.2629])
                img = img * std + mean
                
                # Clip values to valid range
                img = np.clip(img, 0, 1)
            
            # Plot in the appropriate subplot
            axes[task_id, j].imshow(img)
            axes[task_id, j].set_title(f"Class {label}")
            axes[task_id, j].axis('off')
        
        # Add task label to the first image
        axes[task_id, 0].set_ylabel(f"Task {task_id+1}", fontsize=12)
    
    plt.suptitle("Examples from Each Task", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig("task_examples.png")
    plt.show()

def visualize_dataset():
    """
    Load and visualize the GTSRB dataset
    """
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])
    
    # Load GTSRB dataset
    train_dataset = torchvision.datasets.GTSRB(
        root='./data', split='train', download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.GTSRB(
        root='./data', split='test', download=True, transform=transform
    )
    
    # Wrap datasets to add targets attribute
    train_dataset = GTSRBWrapper(train_dataset)
    test_dataset = GTSRBWrapper(test_dataset)
    
    # Create task splits
    train_tasks = create_task_splits(train_dataset, num_tasks=5)
    test_tasks = create_task_splits(test_dataset, num_tasks=5)
    
    print(f"Train dataset size: {len(train_dataset)} images")
    print(f"Test dataset size: {len(test_dataset)} images")
    
    # Plot sample images
    plot_sample_images(train_dataset, num_images=16, title="Sample Training Images from GTSRB")
    
    # Plot class distribution
    plot_class_distribution(train_dataset, title="Class Distribution in GTSRB Training Set")
    
    # Plot examples by class
    plot_class_examples(train_dataset, num_classes=10, samples_per_class=5)
    
    # Plot examples from each task
    plot_task_examples(train_tasks, num_samples=5)
    
    return train_dataset, test_dataset, train_tasks, test_tasks
