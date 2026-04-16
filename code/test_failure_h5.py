import h5py
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import importlib
from tqdm import tqdm
import json

def load_data_hdf5(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'modelnet40_ply_hdf5_2048')
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
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_point', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--model_path', type=str, default='../best_model.pth')
    args = parser.parse_args()

    # Load data
    print("Loading HDF5 data...")
    test_data, test_label = load_data_hdf5('test')
    test_set = ModelNetH5Dataset(test_data, test_label, args.num_point)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Category names
    cat_file = os.path.join('data', 'modelnet40_ply_hdf5_2048', 'shape_names.txt')
    with open(cat_file, 'r') as f:
        cat_names = [line.strip() for line in f.readlines()]

    # Load model
    sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
    model_module = importlib.import_module('pointnet_cls')
    classifier = model_module.get_model(40, normal_channel=False)
    if not args.use_cpu:
        classifier = classifier.cuda()
    
    checkpoint = torch.load(args.model_path, map_location='cpu')
    classifier.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    classifier.eval()

    failure_cases = []
    confusion_matrix = np.zeros((40, 40), dtype=int)
    
    print("\nCollecting failure cases...")
    for j, (points, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
        target = target.squeeze()
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        with torch.no_grad():
            pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        
        for i in range(len(target)):
            true_label = target[i].item()
            pred_label = pred_choice[i].item()
            confusion_matrix[true_label][pred_label] += 1
            if true_label != pred_label:
                failure_cases.append({
                    'index': j * args.batch_size + i,
                    'true': cat_names[true_label],
                    'pred': cat_names[pred_label]
                })

    with open('failure_cases.txt', 'w') as f:
        f.write(f"Total test samples: {len(test_set)}\n")
        f.write(f"Total failures: {len(failure_cases)}\n")
        f.write(f"Overall Accuracy: {(len(test_set) - len(failure_cases)) / len(test_set):.4f}\n\n")
        
        f.write("--- Common Confusion Pairs ---\n")
        confusions = []
        for i in range(40):
            for j in range(40):
                if i != j and confusion_matrix[i][j] > 0:
                    confusions.append((i, j, confusion_matrix[i][j]))
        confusions.sort(key=lambda x: x[2], reverse=True)
        for i, j, count in confusions[:15]:
            f.write(f"{cat_names[i]} -> {cat_names[j]}: {count} cases\n")
        
        f.write("\n--- Failure Cases List ---\n")
        for case in failure_cases[:20]:  # Limit list
            f.write(f"Index: {case['index']}, True: {case['true']}, Pred: {case['pred']}\n")

if __name__ == '__main__':
    import sys
    main()
