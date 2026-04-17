import h5py
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import importlib
from tqdm import tqdm
import sys

# ModelNet10 classes within ModelNet40
MODELNET10_CLASSES = [
    'bathtub', 'bed', 'chair', 'desk', 'dresser', 
    'monitor', 'night_stand', 'sofa', 'table', 'toilet'
]

def load_data_hdf5_subset(partition, target_classes):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'modelnet40_ply_hdf5_2048')
    
    # Get all class names
    with open(os.path.join(DATA_DIR, 'shape_names.txt'), 'r') as f:
        all_cat_names = [line.strip() for line in f.readlines()]
    
    # Find indices of target classes
    target_indices = [all_cat_names.index(cat) for cat in target_classes if cat in all_cat_names]
    
    all_data = []
    all_label = []
    file_list = os.path.join(DATA_DIR, f'{partition}_files.txt')
    with open(file_list, 'r') as f:
        files = [line.strip() for line in f.readlines()]
    
    for h5_name in files:
        h5_path = os.path.join(DATA_DIR, os.path.basename(h5_name))
        f = h5py.File(h5_path, 'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        
        # Filter only target classes
        mask = np.isin(label.squeeze(), target_indices)
        if np.any(mask):
            all_data.append(data[mask])
            # Re-map labels to 0-9 for ModelNet10 evaluation if needed, 
            # but here we keep original indices to test ModelNet40 model
            all_label.append(label[mask])
            
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label, target_indices

class ModelNetH5Dataset(Dataset):
    def __init__(self, data, label, num_points=1024):
        self.data = data
        self.label = label
        self.num_points = num_points

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

def validate(model, loader, use_cpu=False):
    mean_correct = []
    model.eval()
    for points, target in tqdm(loader, total=len(loader), leave=False):
        target = target.squeeze()
        if not use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        with torch.no_grad():
            pred, _ = model(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))
    return np.mean(mean_correct)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_point', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--model_path', type=str, default='../best_model.pth')
    args = parser.parse_args()

    print(f"Testing ModelNet40-trained model on ModelNet10 subset classes...")
    
    # Load data
    test_data, test_label, target_indices = load_data_hdf5_subset('test', MODELNET10_CLASSES)
    test_set = ModelNetH5Dataset(test_data, test_label, args.num_point)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    print(f"Loaded {len(test_set)} samples from 10 classes: {MODELNET10_CLASSES}")

    # Load model
    sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
    model_module = importlib.import_module('pointnet_cls')
    classifier = model_module.get_model(40, normal_channel=False)
    if not args.use_cpu:
        classifier = classifier.cuda()
    
    checkpoint = torch.load(args.model_path, map_location='cpu')
    classifier.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    classifier.eval()

    acc = validate(classifier, test_loader, use_cpu=args.use_cpu)
    print(f"\nModelNet10 Subset Accuracy: {acc:.4f}")
    
    with open('eval_cross_dataset.txt', 'w') as f:
        f.write(f"Dataset: ModelNet10 (Subset of ModelNet40)\n")
        f.write(f"Number of samples: {len(test_set)}\n")
        f.write(f"Accuracy: {acc:.4f}\n")

if __name__ == '__main__':
    main()
