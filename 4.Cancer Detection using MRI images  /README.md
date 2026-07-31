Breast Ultrasound Image Classification (BUSI Dataset)

Overview
This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify breast ultrasound images from the BUSI (Breast Ultrasound Images) dataset. The model categorizes images into three classes: normal, benign, and malignant.

Dataset
Source: Dataset_BUSI_with_GT
Classes: 3 (Normal, Benign, Malignant)
Format: Ultrasound images with corresponding ground truth masks
Preprocessing: Images are resized to 128×128 pixels and normalized (rescaled to 1/255)

Model Architecture
| Layer | Type         | Details                       |
| ----- | ------------ | ----------------------------- |
| 1     | Input        | 128 × 128 × 3                 |
| 2     | Conv2D       | 32 filters, 3×3 kernel, ReLU  |
| 3     | MaxPooling2D | 2×2 pool size                 |
| 4     | Conv2D       | 64 filters, 3×3 kernel, ReLU  |
| 5     | MaxPooling2D | 2×2 pool size                 |
| 6     | Conv2D       | 128 filters, 3×3 kernel, ReLU |
| 7     | MaxPooling2D | 2×2 pool size                 |
| 8     | Flatten      | —                             |
| 9     | Dense        | 128 units, ReLU               |
| 10    | Dense        | 3 units, Softmax (output)     |

Training Details
Framework: TensorFlow/Keras
Optimizer: Adam
Loss Function: Categorical Crossentropy
Metrics: Accuracy
Batch Size: 32
Epochs: 10
Validation Split: 20%
Data Augmentation: Rescaling only (no augmentation applied)

File Structure
plain
/content/
├── Dataset_BUSI_with_GT/        # Extracted dataset
│   ├── benign/
│   ├── malignant/
│   └── normal/
└── Cancer_Detection_CNN.h5      # Saved trained model

Requirements
txt
tensorflow>=2.0
numpy
matplotlib (optional, for visualization)
Usage
1. Upload & Extract Dataset
Python
from google.colab import files
uploaded = files.upload()

import zipfile
with zipfile.ZipFile("dataset.zip", "r") as zip_ref:
    zip_ref.extractall("/content")
2. Load Data
Python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_datagen.flow_from_directory(
    "/content/Dataset_BUSI_with_GT",
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

validation_data = train_datagen.flow_from_directory(
    "/content/Dataset_BUSI_with_GT",
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)
3. Train Model
Python
history = model.fit(train_data, epochs=10, validation_data=validation_data)
4. Save & Load Model
Python
model.save("Cancer_Detection_CNN.h5")

# To load later:
from tensorflow.keras.models import load_model
model = load_model("Cancer_Detection_CNN.h5")
Notes
The model is trained on RGB images (3 channels) despite ultrasound being grayscale — ensure your dataset images are converted to RGB if needed.
The code runs model.fit() twice (20 total epochs). You may want to adjust this based on convergence.
For better performance, consider adding Dropout, Batch Normalization, or using Transfer Learning (e.g., VGG16, ResNet50).
Potential Improvements
[ ] Add data augmentation (rotation, zoom, flip)
[ ] Implement early stopping and learning rate scheduling
[ ] Use pre-trained models (Transfer Learning)
[ ] Add class weights to handle class imbalance
[ ] Evaluate with confusion matrix and classification report
[ ] Implement Grad-CAM for model interpretability

License
This project uses the publicly available BUSI dataset for educational and research purposes. 
