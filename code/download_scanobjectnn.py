import kagglehub
import os
import shutil

# Download latest version
print("Downloading ScanObjectNN-XT from Kaggle...")
try:
    path = kagglehub.dataset_download("ssfailearning/scanobjectnn-xt")
    print("Path to dataset files:", path)

    # Target directory
    target_dir = r"d:\5182CG\PartB\Point_cloud_shape_classification\code\data\ScanObjectNN"
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

    print("ScanObjectNN deployment complete.")
except Exception as e:
    print(f"Error during download/deployment: {e}")
