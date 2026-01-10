import os
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import pil_loader

data_cat = ['train', 'valid'] # data categories

def get_study_level_data(study_type):
    """
    Returns a dict, with keys 'train' and 'valid' and respective values as study level dataframes, 
    these dataframes contain three columns 'Path', 'Count', 'Label'
    Args:
        study_type (string): one of the seven study type folder names in 'train/valid/test' dataset 
    """
    study_data = {}
    study_label = {'positive': 1, 'negative': 0}
    for phase in data_cat:
        # Try MURA, MURA-v1.1, then MURA-v1.0
        base_dir_mura = 'MURA/%s/%s/' % (phase, study_type)
        base_dir_v11 = 'MURA-v1.1/%s/%s/' % (phase, study_type)
        base_dir_v10 = 'MURA-v1.0/%s/%s/' % (phase, study_type)
        
        if os.path.exists(base_dir_mura):
            BASE_DIR = base_dir_mura
        elif os.path.exists(base_dir_v11):
            BASE_DIR = base_dir_v11
        elif os.path.exists(base_dir_v10):
            BASE_DIR = base_dir_v10
        else:
            raise FileNotFoundError(f"MURA dataset not found for {phase}/{study_type}. Checked: MURA/, MURA-v1.1/, MURA-v1.0/")
        patients = list(os.walk(BASE_DIR))[0][1] # list of patient folder names
        study_data[phase] = pd.DataFrame(columns=['Path', 'Count', 'Label'])
        i = 0
        for patient in tqdm(patients): # for each patient folder
            for study in os.listdir(BASE_DIR + patient): # for each study in that patient folder
                label = study_label[study.split('_')[1]] # get label 0 or 1
                path = BASE_DIR + patient + '/' + study + '/' # path to this study
                # Count only image files (png, jpg, jpeg)
                image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if len(image_files) > 0:  # Only add if there are images
                    study_data[phase].loc[i] = [path, len(image_files), label] # add new row
                    i+=1
    return study_data

class ImageDataset(Dataset):
    """training dataset."""

    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): a pandas DataFrame with image path and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        study_path = self.df.iloc[idx, 0]
        count = self.df.iloc[idx, 1]
        images = []
        
        # Get actual image files from directory
        try:
            image_files = sorted([f for f in os.listdir(study_path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            for img_file in image_files:
                try:
                    image = pil_loader(os.path.join(study_path, img_file))
                    images.append(self.transform(image))
                except Exception as e:
                    # Skip corrupted images
                    print(f"Warning: Could not load {os.path.join(study_path, img_file)}: {e}")
                    continue
            
            if len(images) == 0:
                # If no images could be loaded, create a dummy image
                print(f"Warning: No valid images in {study_path}, skipping...")
                raise ValueError(f"No valid images in study: {study_path}")
            
            images = torch.stack(images)
            label = self.df.iloc[idx, 2]
            sample = {'images': images, 'label': label}
            return sample
            
        except Exception as e:
            print(f"Error loading study {study_path}: {e}")
            raise

def get_dataloaders(data, batch_size=8, study_level=False):
    '''
    Returns dataloader pipeline with data augmentation
    '''
    data_transforms = {
        'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: ImageDataset(data[x], transform=data_transforms[x]) for x in data_cat}
    # Set num_workers=0 for Windows compatibility
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in data_cat}
    return dataloaders

if __name__=='main':
    pass
