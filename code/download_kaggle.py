import kagglehub
import os
import shutil

# Download latest version
print("Downloading ModelNet40 (Normal Resampled) from Kaggle...")
path = kagglehub.dataset_download("chenxaoyu/modelnet-normal-resampled")
print("Path to dataset files:", path)

# Target directory
target_dir = r"d:\5182CG\PartB\code\data\modelnet40_normal_resampled"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

print(f"Moving files to {target_dir}...")
# Move everything from download path to target_dir
for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(target_dir, item)
    if os.path.isdir(s):
        if os.path.exists(d):
            shutil.rmtree(d)
        shutil.copytree(s, d)
    else:
        shutil.copy2(s, d)

print("Dataset deployment complete.")
