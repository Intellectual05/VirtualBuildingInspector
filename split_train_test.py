# this file is for splitting images into training and testing sets

import os, shutil
from sklearn.model_selection import train_test_split

def split_data(base_dir, output_dir, test_size=0.1):
    classes = ['Low', 'Medium', 'High']
    for cls in classes:
        images = os.listdir(os.path.join(base_dir, cls))
        train_imgs, val_imgs = train_test_split(images, test_size=test_size, random_state=42)

        for phase in ['train', 'val']:
            os.makedirs(os.path.join(output_dir, phase, cls), exist_ok=True)

        print(f"Adding images from class {cls} to the training dataset.")
        for img in train_imgs:
            src = os.path.join(base_dir, cls, img)
            dst = os.path.join(output_dir, 'train', cls, img)
            shutil.copy(src, dst)

        print(f"Adding images from class {cls} to the testing / validation dataset.")
        for img in val_imgs:
            src = os.path.join(base_dir, cls, img)
            dst = os.path.join(output_dir, 'val', cls, img)
            shutil.copy(src, dst)

split_data("AssignedSeriousness", "yoloDataset")
print("Operation complete!")
