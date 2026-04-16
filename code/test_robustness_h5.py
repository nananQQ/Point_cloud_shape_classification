import h5py
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import importlib
from tqdm import tqdm

def load_data_hdf5(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'modelnet40_ply_hdf5_2048')
    all_data = []
    all_label = []
    file_list = os.path.join(DATA_DIR, f'{partition}_files.txt')
    with open(file_list, 'r') as f:
        files = [line.strip() for line in f.readlines()]
    
    for h5_name in files:
        # The file names in the txt might be relative or full paths, adjust accordingly
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

def add_gaussian_noise(points, sigma=0.01):
    noise = np.random.normal(0, sigma, points.shape)
    return points + noise

def random_point_dropout(points, dropout_ratio=0.1):
    # points: [B, N, C]
    B, N, C = points.shape
    new_points = np.copy(points)
    for i in range(B):
        keep_idx = np.random.choice(N, int(N * (1 - dropout_ratio)), replace=False)
        temp = points[i, keep_idx, :]
        # Fill up to N by duplicating
        fill_idx = np.random.choice(len(keep_idx), N - len(keep_idx), replace=True)
        new_points[i] = np.concatenate([temp, temp[fill_idx]], axis=0)
    return new_points

def validate(model, loader, noise_sigma=0, dropout_ratio=0, use_cpu=False):
    mean_correct = []
    model.eval()
    for points, target in tqdm(loader, total=len(loader), leave=False):
        points = points.numpy()
        if noise_sigma > 0:
            points = add_gaussian_noise(points, noise_sigma)
        if dropout_ratio > 0:
            points = random_point_dropout(points, dropout_ratio)
        
        points = torch.Tensor(points)
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

    # Load data
    print("Loading HDF5 data...")
    test_data, test_label = load_data_hdf5('test')
    test_set = ModelNetH5Dataset(test_data, test_label, args.num_point)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Load model
    sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
    model_module = importlib.import_module('pointnet_cls')
    classifier = model_module.get_model(40, normal_channel=False)
    if not args.use_cpu:
        classifier = classifier.cuda()
    
    checkpoint = torch.load(args.model_path, map_location='cpu')
    classifier.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    classifier.eval()

    results = []
    print("\n--- Testing Noise Robustness ---")
    for sigma in [0, 0.01, 0.02, 0.05]:
        acc = validate(classifier, test_loader, noise_sigma=sigma, use_cpu=args.use_cpu)
        print(f"Sigma: {sigma}, Accuracy: {acc:.4f}")
        results.append(f"noise_sigma_{sigma}: {acc:.4f}")

    print("\n--- Testing Point Dropout Robustness ---")
    for drop in [0, 0.1, 0.3, 0.5]:
        acc = validate(classifier, test_loader, dropout_ratio=drop, use_cpu=args.use_cpu)
        print(f"Dropout: {drop}, Accuracy: {acc:.4f}")
        results.append(f"dropout_{drop}: {acc:.4f}")

    with open('eval_robustness.txt', 'w') as f:
        f.write('\n'.join(results))

if __name__ == '__main__':
    import sys
    main()
