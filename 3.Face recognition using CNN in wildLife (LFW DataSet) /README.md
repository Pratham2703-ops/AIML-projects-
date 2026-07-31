Face Detection with OpenCV
A simple face detection application using OpenCV's Haar Cascade classifier. This project captures images via webcam and detects faces in real-time, drawing bounding boxes around detected faces.

Features
 Webcam Image Capture - Take photos directly from your webcam
 Real-time Face Detection - Detect faces using Haar Cascade classifier
 Bounding Box Visualization - Draws green rectangles around detected faces
 Labeling - Adds "Face Detected" text labels above bounding boxes
 Image Enhancement - Uses histogram equalization for improved detection in varying lighting
 
Requirements
plain
python >= 3.6
opencv-python
numpy

Installation
bash
pip install opencv-python

Project Structure
plain
face-detection/
├── face_detection.py          # Main script
├── photo.jpg                  # Captured image (generated at runtime)
└── README.md                  # This file

Usage
For Jupyter/Colab Notebooks
The script is designed to run in Google Colab or Jupyter environments with webcam access:
Python
# Run the complete script in a notebook cell
# It will:
# 1. Request webcam permission
# 2. Show a "Capture" button
# 3. Take a photo when clicked
# 4. Process and display the image with detected faces

For Local Python Scripts
If running locally, replace the webcam capture section with:
Python
# Instead of take_photo(), use an existing image
image = cv2.imread("your_image.jpg")

How It Works
1. Image Capture
Uses JavaScript to access the browser's webcam API
Displays a live video feed with a capture button
Saves the captured frame as a JPEG file
2. Face Detection Pipeline
plain
Original Image → Grayscale Conversion → Histogram Equalization → 
Haar Cascade Detection → Draw Bounding Boxes → Display Result
3. Detection Parameters
Table
Parameter	Value	Description
scaleFactor	1.1	How much the image size is reduced at each scale
minNeighbors	3	Minimum number of rectangles to retain a detection
minSize	(30,30)	Minimum possible object size

Code Explanation
Key Components
Face Detector Initialization
Python
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

Image Preprocessing
Python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
gray = cv2.equalizeHist(gray)                    # Enhance contrast
Face Detection
Python
faces = face_detector.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=3,
    minSize=(30, 30)
)
Drawing Results
Python
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image, "Face Detected", (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
Error Handling
The improved version includes checks for:
 Haar Cascade file loading status
 Image file existence and loading
 Empty detection results handling
 
Customization
Adjust Detection Sensitivity
Table
Goal	Action
Detect more faces	Lower minNeighbors to 2-3
Reduce false positives	Increase minNeighbors to 5-6
Detect smaller faces	Lower minSize or scaleFactor
Detect only large faces	Increase minSize
Change Visual Style
Python
# Different colors (BGR format)
RED    = (0, 0, 255)
BLUE   = (255, 0, 0)
YELLOW = (0, 255, 255)

# Change rectangle thickness
cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)  # Thicker

# Change font size
cv2.putText(image, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
Limitations
Requires frontal face orientation (profile detection needs different cascade)
Performance depends on lighting conditions
May produce false positives with face-like patterns
Webcam access requires browser permissions (Colab/Jupyter only)
Troubleshooting
Table
Issue	Solution
"Haar Cascade file not loaded"	Reinstall opencv-python: pip install --force-reinstall opencv-python
"Image not found"	Ensure webcam permission is granted
No faces detected	Improve lighting, face the camera directly
Too many false positives	Increase minNeighbors value
Future Enhancements
[ ] Add support for multiple face detection algorithms (DNN, MTCNN)
[ ] Implement real-time video stream processing
[ ] Add face recognition capabilities
[ ] Support for side-profile face detection
[ ] Save detected faces as separate cropped images
License
This project is open-source. Feel free to use and modify as needed.
Acknowledgments
OpenCV Library: https://opencv.org/
Haar Cascade Classifiers: Pre-trained models by OpenCV community
