import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
from data_utils.ModelNetDataLoader import ModelNetDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing Robustness')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--model_path', type=str, default='../best_model.pth', help='Model path')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()

def add_gaussian_noise(points, sigma=0.01):
    """ Add Gaussian noise to point cloud """
    noise = np.random.normal(0, sigma, points.shape)
    return points + noise

def random_point_dropout(points, dropout_ratio=0.1):
    """ Randomly dropout points """
    # points: [B, N, C] or [N, C]
    if len(points.shape) == 3:
        B, N, C = points.shape
        new_points = np.zeros(points.shape)
        for i in range(B):
            keep_idx = np.random.choice(N, int(N * (1 - dropout_ratio)), replace=False)
            # Fill with first point or mean to keep same shape if needed, 
            # but usually PointNet handles fixed N. 
            # To keep N constant, we can duplicate remaining points.
            new_points[i, :len(keep_idx), :] = points[i, keep_idx, :]
            if len(keep_idx) < N:
                # Duplicate points to fill up to N
                fill_idx = np.random.choice(keep_idx, N - len(keep_idx), replace=True)
                new_points[i, len(keep_idx):, :] = points[i, fill_idx, :]
        return new_points
    else:
        N, C = points.shape
        keep_idx = np.random.choice(N, int(N * (1 - dropout_ratio)), replace=False)
        new_points = points[keep_idx, :]
        if len(keep_idx) < N:
            fill_idx = np.random.choice(keep_idx, N - len(keep_idx), replace=True)
            new_points = np.concatenate([new_points, points[fill_idx, :]], axis=0)
        return new_points

def validate(model, loader, noise_sigma=0, dropout_ratio=0, num_class=40):
    mean_correct = []
    model.eval()
    
    for j, (points, target) in tqdm(enumerate(loader), total=len(loader), leave=False):
        points = points.numpy()
        if noise_sigma > 0:
            points = add_gaussian_noise(points, noise_sigma)
        if dropout_ratio > 0:
            points = random_point_dropout(points, dropout_ratio)
        
        points = torch.Tensor(points)
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        with torch.no_grad():
            pred, _ = model(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    instance_acc = np.mean(mean_correct)
    return instance_acc

def main():
    global args
    args = parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Load Data
    data_path = 'data/modelnet40_normal_resampled/'
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} not found.")
        # Try parent directory data path
        data_path = '../data/modelnet40_normal_resampled/'
        if not os.path.exists(data_path):
            print(f"Error: Data path {data_path} not found.")
            # return
    
    print('Loading dataset...')
    try:
        test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
        testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    except Exception as e:
        print(f"Failed to load data: {e}")
        # Mock data for demonstration if data is missing (not ideal but for completeness)
        # return

    # Load Model
    num_class = args.num_category
    model_module = importlib.import_module('pointnet_cls')
    classifier = model_module.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    print(f"Loading checkpoint from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        classifier.load_state_dict(checkpoint)

    results = []
    
    # 1. Noise Robustness
    print("\n--- Testing Noise Robustness ---")
    sigmas = [0, 0.01, 0.02, 0.05]
    for sigma in sigmas:
        acc = validate(classifier, testDataLoader, noise_sigma=sigma, num_class=num_class)
        print(f"Sigma: {sigma}, Accuracy: {acc:.4f}")
        results.append(f"noise_sigma_{sigma}: {acc:.4f}")

    # 2. Dropout Robustness
    print("\n--- Testing Point Dropout Robustness ---")
    dropouts = [0, 0.1, 0.3, 0.5]
    for drop in dropouts:
        acc = validate(classifier, testDataLoader, dropout_ratio=drop, num_class=num_class)
        print(f"Dropout: {drop}, Accuracy: {acc:.4f}")
        results.append(f"dropout_{drop}: {acc:.4f}")

    # Save to file
    with open('eval_robustness.txt', 'w') as f:
        for res in results:
            f.write(res + '\n')
    print("\nResults saved to eval_robustness.txt")

if __name__ == '__main__':
    main()
