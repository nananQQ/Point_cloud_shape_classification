import h5py
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import importlib
from tqdm import tqdm
import sys

# Common classes between ModelNet40 and ScanObjectNN (Approximate mapping)
# ModelNet40 categories (40): airplane, bathtub, bed, bench, bookshelf, bottle, bowl, car, chair, cone, cup, curtain, desk, door, dresser, flower_pot, glass_box, guitar, keyboard, lamp, laptop, mantel, monitor, night_stand, person, piano, plant, radio, range_hood, sink, sofa, stairs, stool, table, tent, toilet, tv_stand, vase, wardrobe, xbox
# ScanObjectNN categories (15): bag, bin, box, cabinet, chair, desk, display, door, shelf, table, bed, sofa, sink, toilet, sofa

SCANOBJECTNN_TO_MODELNET40 = {
    'bed': 'bed',
    'chair': 'chair',
    'desk': 'desk',
    'door': 'door',
    'display': 'monitor',
    'shelf': 'bookshelf',
    'sofa': 'sofa',
    'table': 'table',
    'sink': 'sink',
    'toilet': 'toilet',
    'tub': 'bathtub',
    'cabinet': 'wardrobe'
}

# ScanObjectNN standard class names (PB_T50_RS split often uses these)
SCANOBJECTNN_CLASSES = [
    'bag', 'bin', 'box', 'cabinet', 'chair', 
    'desk', 'display', 'door', 'shelf', 'table', 
    'bed', 'sofa', 'sink', 'toilet', 'tub'
]

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def load_scanobjectnn_data(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'ScanObjectNN')
    
    # Try to find the h5 file. ScanObjectNN-XT often has h5_files/main_split/test_objectdataset.h5
    h5_path = os.path.join(DATA_DIR, 'h5_files', 'main_split', f'{partition}_objectdataset.h5')
    if not os.path.exists(h5_path):
        # Alternative path
        h5_path = os.path.join(DATA_DIR, f'{partition}_objectdataset.h5')
        if not os.path.exists(h5_path):
            # Try specific augmented names seen in directory
            aug_name = 'training' if partition == 'train' else 'test'
            h5_path = os.path.join(DATA_DIR, f'{aug_name}_objectdataset_augmentedrot_scale75.h5')
            if not os.path.exists(h5_path):
                print(f"Error: Could not find ScanObjectNN h5 file at {h5_path}")
                return None, None

    f = h5py.File(h5_path, 'r')
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    mask = f['mask'][:] if 'mask' in f else None
    f.close()
    return data, label, mask

class ScanObjectNNDataset(Dataset):
    def __init__(self, data, label, mask=None, num_points=1024):
        self.data = data
        self.label = label
        self.mask = mask
        self.num_points = num_points

    def __getitem__(self, item):
        pointcloud = self.data[item]
        if self.mask is not None:
            # Filter by mask (only object points)
            mask = self.mask[item] > 0
            if mask.sum() > 0:
                pointcloud = pointcloud[mask]
        
        # Take num_points. If not enough, pad or repeat.
        if len(pointcloud) >= self.num_points:
            # Just take first num_points (or could use FPS)
            pointcloud = pointcloud[:self.num_points]
        else:
            # Pad by repeating points
            indices = np.random.choice(len(pointcloud), self.num_points, replace=True)
            pointcloud = pointcloud[indices]

        pointcloud = pc_normalize(pointcloud)
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

def validate(model, loader, use_cpu=False, modelnet_classes=None):
    model.eval()
    
    correct_count = 0
    total_count = 0
    
    class_stats = {} # {class_name: [correct, total]}
    
    for points, target in tqdm(loader, total=len(loader), leave=False):
        target = target.squeeze()
        if not use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        with torch.no_grad():
            pred, _ = model(points)
        pred_choice = pred.data.max(1)[1]
        
        for i in range(len(target)):
            s_label = target[i].item()
            s_class_name = SCANOBJECTNN_CLASSES[s_label]
            
            # Find if this ScanObjectNN class has a ModelNet40 equivalent
            m_class_equivalent = SCANOBJECTNN_TO_MODELNET40.get(s_class_name)

            if m_class_equivalent and m_class_equivalent in modelnet_classes:
                if s_class_name not in class_stats:
                    class_stats[s_class_name] = [0, 0]
                
                m_label_expected = modelnet_classes.index(m_class_equivalent)
                total_count += 1
                class_stats[s_class_name][1] += 1
                
                if pred_choice[i].item() == m_label_expected:
                    correct_count += 1
                    class_stats[s_class_name][0] += 1
                    
    print("\nPer-class Accuracy on Overlapping Classes:")
    for cls, stats in class_stats.items():
        if stats[1] > 0:
            print(f"  {cls:10s}: {stats[0]/stats[1]:.4f} ({stats[0]}/{stats[1]})")
            
    return correct_count / total_count if total_count > 0 else 0, class_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_point', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--model_path', type=str, default='../best_model.pth')
    args = parser.parse_args()

    print(f"Testing ModelNet40-trained model on ScanObjectNN (Real-world Scans)...")
    
    # Load data
    data, label, mask = load_scanobjectnn_data('test')
    if data is None: return
    
    test_set = ScanObjectNNDataset(data, label, mask, args.num_point)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    print(f"Loaded {len(test_set)} samples from ScanObjectNN.")

    # Get ModelNet40 class names for mapping
    DATA_DIR_M40 = os.path.join(os.path.dirname(__file__), 'data', 'modelnet40_ply_hdf5_2048')
    with open(os.path.join(DATA_DIR_M40, 'shape_names.txt'), 'r') as f:
        m40_classes = [line.strip() for line in f.readlines()]

    # Load model
    sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
    model_module = importlib.import_module('pointnet_cls')
    classifier = model_module.get_model(40, normal_channel=False)
    if not args.use_cpu:
        classifier = classifier.cuda()
    
    checkpoint = torch.load(args.model_path, map_location='cpu')
    classifier.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    classifier.eval()

    acc, class_stats = validate(classifier, test_loader, use_cpu=args.use_cpu, modelnet_classes=m40_classes)
    print(f"\nScanObjectNN Generalization Accuracy (on overlapping classes): {acc:.4f}")
    
    with open('eval_scanobjectnn.txt', 'w') as f:
        f.write(f"Dataset: ScanObjectNN-XT (Filtered by mask)\n")
        f.write(f"Number of test samples (total): {len(test_set)}\n")
        f.write(f"Accuracy (overlapping classes): {acc:.4f}\n")
        f.write(f"\nPer-class Accuracy:\n")
        for cls, stats in class_stats.items():
            if stats[1] > 0:
                f.write(f"{cls:10s}: {stats[0]/stats[1]:.4f} ({stats[0]}/{stats[1]})\n")

if __name__ == '__main__':
    main()
