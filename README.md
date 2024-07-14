# Celebrity Face Recognition Using OpenCV

This project utilizes the OpenCV face recognizer to identify and recognize the faces of various celebrities. By leveraging OpenCV's powerful image processing capabilities, the system can accurately match and recognize celebrity faces from a given set of images.

## Features

- **Data Collection**: Compile a dataset of celebrity images, with multiple images for each celebrity to enhance recognition accuracy.
- **Face Detection**: Use OpenCV's pre-trained Haar Cascade classifiers to detect faces in the images. This step isolates the face region from the rest of the image.
- **Feature Extraction**: Process detected faces to extract key features unique to each individual, creating a model for each celebrity.
- **Training the Recognizer**: Train OpenCV's face recognizer, such as the LBPH (Local Binary Patterns Histograms) face recognizer, using the collected dataset. This involves feeding the recognizer with face images and their corresponding labels (celebrity names).
- **Face Recognition**: Recognize and identify faces in new images by comparing detected face features with stored models to find the best match.
