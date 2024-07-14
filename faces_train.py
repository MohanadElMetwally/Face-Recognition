import os
import cv2 as cv
import numpy as np

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
DIR = 'faces/train'

# Initialize features and labels lists
features = []
labels = []

# prepare face detector
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Funtion to fetch the training images from training folder
def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        # find and load all images in the training folder
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            # detect face in the loaded image
            face_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in face_rect:
                # select face region only from the image
                face_roi = gray[y:y+h, x:x+w]

                # append selected region to features
                features.append(face_roi)
                labels.append(label)

create_train()
print(f'{" Creating training set is Done! ":-^50}')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)

print(f'{" Model training is Done! ":-^50}')


np.save('features.npy', features)
np.save('labels.npy', labels)
face_recognizer.save('face_recognizer.yml')




cv.waitKey(0)
