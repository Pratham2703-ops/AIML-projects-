CIFAR-10 Image Classification with CNN
A Convolutional Neural Network (CNN) built with TensorFlow/Keras for classifying images from the CIFAR-10 dataset.
Overview
This project implements a deep learning model to classify 32×32 color images into 10 categories using a CNN architecture with three convolutional layers followed by fully connected layers.
Dataset
CIFAR-10 contains 60,000 32×32 color images across 10 classes:
Table
Class	Description
0	Airplane
1	Automobile
2	Bird
3	Cat
4	Deer
5	Dog
6	Frog
7	Horse
8	Ship
9	Truck
Training set: 50,000 images
Test set: 10,000 images
Model Architecture
plain
Input (32×32×3)
    ↓
Conv2D (32 filters, 3×3) + ReLU
    ↓
MaxPooling2D (2×2)
    ↓
Conv2D (64 filters, 3×3) + ReLU
    ↓
MaxPooling2D (2×2)
    ↓
Conv2D (64 filters, 3×3) + ReLU
    ↓
Flatten
    ↓
Dense (64 units) + ReLU
    ↓
Dense (10 units) + Softmax
    ↓
Output (10 classes)
Total Parameters: ~93,000
Requirements
plain
tensorflow>=2.0
numpy
matplotlib (optional, for visualization)
Install dependencies:
bash
pip install tensorflow numpy matplotlib
Usage
1. Clone and Setup
bash
git clone <repository-url>
cd cifar10-cnn
2. Run Training
bash
python train.py
The script will:
Automatically download CIFAR-10 dataset on first run
Normalize pixel values to [0, 1]
Train the CNN for 10 epochs
Evaluate on test set
Print test accuracy
3. Training Configuration
Table
Parameter	Value
Optimizer	Adam
Loss Function	Sparse Categorical Crossentropy
Metric	Accuracy
Epochs	10
Batch Size	64
Validation Data	Test set (x_test, y_test)
Expected Results
After 10 epochs, the model typically achieves:
Training Accuracy: ~75-80%
Test Accuracy: ~68-72%
Note: Results may vary slightly due to random initialization. For better accuracy, consider adding data augmentation, dropout, or training for more epochs.
File Structure
plain
cifar10-cnn/
├── train.py          # Main training script
├── README.md         # This file
└── models/           # Saved model checkpoints (optional)
Key Features
✅ Automatic dataset download via Keras
✅ Pixel normalization for faster convergence
✅ Model summary display for architecture verification
✅ Validation monitoring during training
✅ Final test evaluation with accuracy metric
Improvements to Consider
Table
Technique	Expected Impact
Data Augmentation (rotation, flip, zoom)	+3-5% accuracy
Dropout Regularization	Reduce overfitting
Batch Normalization	Faster training, better stability
Deeper Network (more conv layers)	Higher capacity
Learning Rate Scheduling	Better convergence
Transfer Learning (ResNet/EfficientNet)	+10-15% accuracy
License
MIT License - feel free to use and modify.
Acknowledgments
CIFAR-10 Dataset by Alex Krizhevsky
TensorFlow and Keras teams
