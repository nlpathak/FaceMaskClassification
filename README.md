# Face Mask Classification via Deep Learning

In the midst of the COVID-19 pandemic, the CDC recommends everyone face masks when in public to prevent spread of the virus. Many states like California have mandated mask usage in “public and workplace settings where there is high risk of exposure”. Hence, it is important to identify individuals who are not or incorrectly wearing masks to reduce viral spread and protect those at higher risk. Our project solves this through a deep learning approach that can classify a face in one of three classes: masked, unmasked, masked incorrectly.

### Python Notebooks
* **pytorch-fasterrcnn.ipynb**: train face detector model and apply on some test images for examples
* **FaceMaskClassification_Custom_and_Resnet.ipynb**: train and test Resnet 18, 34, and custom CNN model for face mask classification + code for confusion matrix and end-to-end color bounding-box algorithm
* **FaceMaskClassification_DenseNet.ipynb**: train and test Densenet 121, 161, 201 models for face mask classification
* **FaceMaskClassification_VGG.ipynb**: train and test VGG-16, VGG-19 models for face mask classification

### Settings
Python 3.7.3
PyTorch 1.4.0
TensorFlow 2.5.0

### How to Run
* Download Face Mask Dataset from Kaggle: https://www.kaggle.com/andrewmvd/face-mask-detection

    Place the extracted dataset project's home directory as follows:
    ```
    ./input/Face_Mask_Dataset/annotations
    ./input/Face_Mask_Dataset/images
    ```
* Run pytorch-fasterrcnn.ipynb to train face detector model
    Model will be used to detect faces before training classification models and are saved to:
    ```
    ./saved_models/rcnn_model.pt
    ```

* Run FaceMaskClassification_VGG.ipynb to train VGG models and visualize loss
  
  Run FaceMaskClassification_DenseNet.ipynb to train DenseNet models and visualize loss
  
  Run FaceMaskClassification_Custom_and_Resnet.ipynb to train Resnet/Custom CNN models and visualize loss. Also displays confusion matrix and end-to-end color bounding-box algorithm on test images.
  
  All models will be saved to ./saved_models as .pt
