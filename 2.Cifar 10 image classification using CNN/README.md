 CIFAR-10 Image Classification using CNN
 Objective
Build a Convolutional Neural Network (CNN) to classify 32×32 color images into 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
 Dataset
Source: CIFAR-10 (Canadian Institute For Advanced Research)
Size: 60,000 images (50,000 train + 10,000 test)
Classes: 10
Image Size: 32 × 32 × 3 (RGB)
🔧 Methodology
Data Preprocessing — Normalize pixel values, data augmentation (rotation, flip, zoom)
Model Architecture — CNN with Conv2D, MaxPooling, Dropout, BatchNorm, Dense layers
Training — Adam optimizer, categorical crossentropy, early stopping
Evaluation — Accuracy, Confusion Matrix, Classification Report
 Results
Table
Metric	Score
Training Accuracy	~95%
Validation Accuracy	~85-88%
Test Accuracy	~85%
How to Run
bash
cd Cifar-10-Image-Classification-CNN/
pip install -r requirements.txt
python train.py
# or open cifar10_cnn.ipynb
 Files
data/ — CIFAR-10 dataset (auto-downloaded via Keras)
cifar10_cnn.ipynb — Full notebook with EDA and training
train.py — Training script
model.h5 — Saved Keras model
requirements.txt — Dependencies
 Key Takeaways
Data augmentation is crucial for improving generalization on small images
Batch Normalization and Dropout help prevent overfitting
Deeper architectures (ResNet-style) can push accuracy above 90% 
