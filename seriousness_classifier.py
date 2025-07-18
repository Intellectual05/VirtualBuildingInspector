#IMPORTANT: TRADITIONAL COMPUTER VISION IMPLEMENTED HERE

import cv2 as cv # type: ignore
import numpy as np # type: ignore
import os
from tqdm import tqdm # type: ignore
import shutil
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.cluster import KMeans # type: ignore

input_folder = r"OriginalDataset\Positive"
output_folder = r"AssignedSeriousness"

# function to extract image features using traditional computer vision methods
def extract_features(image_path):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE) # read the image and parse as grayscale
    gaussian_blur = cv.GaussianBlur(img, (5, 5), 0) # to reduce overall noise
    bilateral_blur = cv.bilateralFilter(gaussian_blur, 9, 75, 75) # to remove impulse noise effectively
    # enhance the contrast for the edges to still be visible by canny detection
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(bilateral_blur)
    edges = cv.Canny(enhanced, 100, 200) # detecting edges using Canny method
    # using Sobel detection to determine the orientation of the edges
    sobelx = cv.Sobel(edges, -1, 1, 0, ksize=5)
    sobely = cv.Sobel(edges, -1, 0, 1, ksize=5)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    angle = np.arctan2(sobely, sobelx) * 180 / np.pi  # Convert to degrees
    angle = np.mod(angle + 90, 180)  # Map angles to [0, 180)
    avg_magnitude = float(np.mean(magnitude))
    avg_angle = float(np.mean(np.abs(angle)))
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # finding contours to get insights of crack width and crack area
    # extracting additional features based on the contours
    total_area = 0
    total_perimeter = 0
    crack_widths = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        perimeter = cv.arcLength(cnt, True)
        x, y, w, h = cv.boundingRect(cnt)
        crack_widths.append(w)  # Use width as proxy
        total_area += area
        total_perimeter += perimeter
    
    avg_area = (total_area / len(contours)) if total_area != 0 else 0
    avg_perimeter = (total_perimeter / len(contours)) if total_perimeter != 0 else 0
    avg_width = np.mean(crack_widths) if crack_widths else 0
    edge_density = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])
    return [avg_magnitude, avg_angle, avg_area, avg_perimeter, avg_width, edge_density] # return the features as a list


# collect the features and image paths
image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
               if f.lower().endswith(('.jpg'))]
features = []
# for each image extract the features and store in nested list
for path in tqdm(image_paths, desc="Extracting features"):
    try:
        feat = extract_features(path)
        features.append(feat)
    except Exception as e:
        print(f"Error processing {path}: {e}")
        continue
    
# Normalize features to perform K-means clustering
features = np.array(features)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

print("Performing K-Means Clustering")
# KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(features_scaled)

# Optional: sort clusters by average edge intensity to map to Low, Medium, High
cluster_order = np.argsort([np.mean(features[labels == i, 0]) for i in range(2)])
label_mapping = {cluster_order[0]: 'Medium', cluster_order[1]: 'High'}

print("Clustering Complete! Classifying data to different serious levels based on cluster data.")
# Move files to respective folders
for img_path, label in zip(image_paths, labels):
    class_folder = label_mapping[label]
    filename = os.path.basename(img_path)
    shutil.copy(img_path, os.path.join(output_folder, class_folder, filename))
    print(filename)
    
print("Process completed!")
