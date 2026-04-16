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
    parser = argparse.ArgumentParser('Failure Analysis')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--model_path', type=str, default='../best_model.pth', help='Model path')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Load Data
    data_path = 'data/modelnet40_normal_resampled/'
    if not os.path.exists(data_path):
        data_path = '../data/modelnet40_normal_resampled/'
    
    print('Loading dataset...')
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Category names
    cat_file = os.path.join(data_path, 'modelnet40_shape_names.txt')
    if not os.path.exists(cat_file):
        # Default category list for ModelNet40
        cat_names = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
    else:
        with open(cat_file, 'r') as f:
            cat_names = [line.strip() for line in f.readlines()]

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

    classifier.eval()
    
    failure_cases = []
    confusion_matrix = np.zeros((num_class, num_class), dtype=int)
    
    print("\nCollecting failure cases...")
    for j, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        with torch.no_grad():
            pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        
        # Check failures
        for i in range(len(target)):
            true_label = target[i].item()
            pred_label = pred_choice[i].item()
            confusion_matrix[true_label][pred_label] += 1
            
            if true_label != pred_label:
                # Store enough info to identify the file if possible
                # ModelNetDataLoader stores datapath in self.datapath
                # j*batch_size + i is the index in dataset
                data_idx = j * args.batch_size + i
                if data_idx < len(test_dataset.datapath):
                    shape_name, file_path = test_dataset.datapath[data_idx]
                    failure_cases.append({
                        'index': data_idx,
                        'true': cat_names[true_label],
                        'pred': cat_names[pred_label],
                        'file': file_path
                    })

    # Save results
    with open('failure_cases.txt', 'w') as f:
        f.write(f"Total test samples: {len(test_dataset)}\n")
        f.write(f"Total failures: {len(failure_cases)}\n")
        f.write(f"Overall Accuracy: {(len(test_dataset) - len(failure_cases)) / len(test_dataset):.4f}\n\n")
        
        f.write("--- Common Confusion Pairs ---\n")
        # Find top confusion pairs
        confusions = []
        for i in range(num_class):
            for j in range(num_class):
                if i != j and confusion_matrix[i][j] > 0:
                    confusions.append((i, j, confusion_matrix[i][j]))
        
        confusions.sort(key=lambda x: x[2], reverse=True)
        for i, j, count in confusions[:15]:
            f.write(f"{cat_names[i]} -> {cat_names[j]}: {count} cases\n")
        
        f.write("\n--- Failure Cases List ---\n")
        for case in failure_cases:
            f.write(f"Index: {case['index']}, True: {case['true']}, Pred: {case['pred']}, File: {case['file']}\n")

    print(f"\nCollected {len(failure_cases)} failure cases. Results saved to failure_cases.txt")

if __name__ == '__main__':
    main()
