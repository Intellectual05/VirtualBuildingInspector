# VirtualBuildingInspector

A computer-vision project for detecting concrete cracks and classifying
images into three indicative severity categories: **Low**, **Medium**,
and **High**. The project combines traditional image processing,
unsupervised feature clustering, YOLOv8 training, and a Flask web
interface for image upload and camera-based inspection.

> **Important limitation:** This is an academic prototype, not a
> certified structural-inspection tool. The source dataset provides
> crack/no-crack labels rather than verified severity labels, so the
> Medium/High categories are inferred from image features and
> clustering. Results should not be used as a substitute for assessment
> by a qualified structural engineer.

## Overview

Concrete cracks can vary in appearance and potential seriousness. This
project explores a computer-vision pipeline that first derives image
features using traditional image processing, then groups crack-positive
images into severity categories, and finally trains a YOLOv8 model to
classify images through a simple web application.

### Severity categories

-   **Low** --- no visible crack detected.
-   **Medium** --- crack image assigned to the lower-severity cluster,
    intended to represent minor cracks such as hairline or settlement
    cracks.
-   **High** --- crack image assigned to the higher-severity cluster,
    intended to represent potentially more serious cracks.

These categories are project-defined estimates, not engineering
diagnoses.

## Pipeline

1.  **Dataset preparation**\
    Uses the public Surface Crack Detection dataset, containing
    crack-positive and crack-negative images. The project documentation
    describes at least 40,000 images, split approximately evenly between
    the two original classes.

2.  **Traditional computer vision with OpenCV**\
    Each image is converted to grayscale and processed using:

    -   Gaussian blur (5 × 5 kernel) to reduce noise
    -   Bilateral filtering to smooth while preserving edges
    -   Canny edge detection
    -   CLAHE contrast enhancement
    -   Sobel gradients to estimate edge magnitude and orientation
    -   Contour and convex-hull analysis to derive geometric features

3.  **Feature extraction**\
    The pipeline calculates features including:

    -   Mean edge magnitude
    -   Mean edge angle
    -   Crack-mask area
    -   Crack-mask perimeter
    -   Mean Canny edge intensity
    -   Mean edge width

4.  **Unsupervised severity grouping**\
    The original negative class is mapped to **Low**. Feature vectors
    from crack-positive images are standardized and grouped into two
    clusters with K-means, which are used to assign **Medium** and
    **High** labels. Because the original dataset does not provide
    verified severity labels, these cluster assignments cannot be
    treated as ground-truth severity assessments.

5.  **YOLOv8 training**\
    The derived labels are used to train a YOLOv8 model. The project
    documentation reports training for 30 epochs with a 227-pixel image
    dimension and approximately 90% training accuracy. This figure is a
    reported training result, not a validated estimate of real-world
    performance.

6.  **Web application**\
    A Flask backend connects the trained model to an HTML interface.
    Users can upload an image or capture one with a device camera; the
    application returns a predicted severity category and a confidence
    value.

## Repository structure

``` text
VirtualBuildingInspector/
├── application/              # Flask application and web interface
├── crack_severity.yaml        # YOLO task/class configuration
├── seriousness_classifier.py  # Severity classification / feature pipeline
├── split_train_test.py        # Dataset splitting utility
└── README.md
```

## Technologies

-   **Python**
-   **OpenCV** --- image preprocessing, edge detection, gradient and
    contour features
-   **scikit-learn** --- feature standardization and K-means clustering
-   **YOLOv8 / Ultralytics** --- image-model training and inference
-   **Flask** --- web application backend
-   **HTML** --- browser interface

## Dataset

The project uses the [Surface Crack Detection dataset on
Kaggle](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection).

The dataset's original labels indicate whether a crack is present; they
do not provide verified Low/Medium/High severity labels. The severity
pipeline therefore creates approximate labels from extracted image
features and clustering.

## Training notes

The project documentation records the following experimental settings:

-   YOLOv8 training for **30 epochs**
-   Image dimension: **227 pixels**
-   Approximately **90% reported training accuracy**
-   GPU computation used for training

The documentation describes a train/validation split, but its text
contains an inconsistency in the stated proportions. Confirm the actual
split in the current training code before relying on a particular ratio.

## Running the web application

1.  Clone or download this repository.

2.  Install the Python dependencies required by the application and
    model code.

3.  Open a terminal and change to the `application` directory.

4.  Run:

    ``` bash
    python app.py
    ```

5.  Open the local URL printed in the terminal (the project
    documentation gives `http://127.0.0.1:5000`).

6.  Upload an image, or allow browser camera access to capture a
    photograph.

7.  Stop the local server with `Ctrl+C`.

Dependency versions and model-weight paths are not specified here
because they should match the files currently present in the repository.
If the application fails to start, check the imports, installed
packages, and configured path to the trained model.

## Evaluation and limitations

The project was tested on unseen crack images, and the application was
reported to return Medium or High predictions for crack-positive
examples. However:

-   The source dataset does not include verified severity labels.
-   K-means cluster quality therefore cannot be evaluated directly
    against ground-truth severity.
-   Lighting and image brightness affect feature extraction and may lead
    to misclassification.
-   The reported training accuracy should not be interpreted as test
    accuracy, clinical/engineering validation, or a measure of
    structural safety.
-   A false negative on a serious crack could have significant
    consequences; all findings require professional inspection.

## Training notebook

OpenCV experimentation and YOLO training were also performed in Google
Colab:

[Open the Colab
notebook](https://colab.research.google.com/drive/1eEocbpZjy4k0-AMTIW4yD3LI40vo4WXA?usp=sharing)

## Author

**Agamjit Singh**

Developed as a computer-vision project exploring traditional image
processing, unsupervised feature grouping, deep-learning-based
classification, and web deployment for preliminary concrete crack
inspection.
