"""
Fixed training script for MURA - compatible with modern PyTorch
"""

import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import numpy as np

data_cat = ['train', 'valid']

class ConfusionMeter:
    """Simple confusion matrix meter"""
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.preds = []
        self.labels = []
    
    def add(self, preds, labels):
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        preds = preds.flatten()
        labels = labels.flatten()
        
        self.preds.extend(preds)
        self.labels.extend(labels)
    
    def value(self):
        if len(self.preds) == 0:
            return np.zeros((self.num_classes, self.num_classes))
        cm = sk_confusion_matrix(self.labels, self.preds, labels=list(range(self.num_classes)))
        # Normalize
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
        return cm

def train_model(model, criterion, optimizer, dataloaders, scheduler, 
                dataset_sizes, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    costs = {x:[] for x in data_cat}
    accs = {x:[] for x in data_cat}
    prev_lr = optimizer.param_groups[0]['lr']
    device = next(model.parameters()).device
    
    print('Train batches:', len(dataloaders['train']))
    print('Valid batches:', len(dataloaders['valid']), '\n')
    
    for epoch in range(num_epochs):
        confusion_matrix = {x: ConfusionMeter(2) for x in data_cat}
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        
        for phase in data_cat:
            model.train(phase=='train')
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0
            
            for i, data in enumerate(dataloaders[phase]):
                if i % 100 == 0:
                    print(f'  Batch {i}/{len(dataloaders[phase])}', end='\r')
                
                inputs = data['images'][0]
                labels = data['label'].type(torch.FloatTensor)
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                outputs = torch.mean(outputs)
                loss = criterion(outputs, labels, phase)
                
                running_loss += loss.item()
                total_samples += labels.size(0)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                preds = (outputs > 0.5).float()
                running_corrects += torch.sum(preds == labels).item()
                confusion_matrix[phase].add(preds.cpu(), labels.cpu())
            
            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / total_samples if total_samples > 0 else 0.0
            
            costs[phase].append(epoch_loss)
            accs[phase].append(epoch_acc)
            
            print(f'{phase:6s} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            cm = confusion_matrix[phase].value()
            print(f'Confusion Matrix:\n{cm}')
            
            if phase == 'valid':
                scheduler.step(epoch_loss)
                current_lr = optimizer.param_groups[0]['lr']
                if epoch == 0 or abs(current_lr - prev_lr) > 1e-6:
                    print(f'Learning rate: {current_lr:.6f}')
                    prev_lr = current_lr
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        
        time_elapsed = time.time() - since
        print('Time elapsed: {:.0f}m {:.0f}s\n'.format(
                time_elapsed // 60, time_elapsed % 60))
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:.4f}'.format(best_acc))
    
    model.load_state_dict(best_model_wts)
    return model

def get_metrics(model, criterion, dataloaders, dataset_sizes, phase='valid'):
    """Get metrics for a phase"""
    confusion_matrix = ConfusionMeter(2)
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloaders[phase]):
            if i % 50 == 0:
                print(f'  Batch {i}/{len(dataloaders[phase])}', end='\r')
            
            labels = data['label'].type(torch.FloatTensor)
            inputs = data['images'][0]
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            outputs = torch.mean(outputs)
            loss = criterion(outputs, labels, phase)
            
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            preds = (outputs > 0.5).float()
            running_corrects += torch.sum(preds == labels).item()
            confusion_matrix.add(preds.cpu(), labels.cpu())
    
    loss = running_loss / total_samples if total_samples > 0 else 0.0
    acc = running_corrects / total_samples if total_samples > 0 else 0.0
    
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, loss, acc))
    cm = confusion_matrix.value()
    print('Confusion Matrix:\n{}'.format(cm))

