import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_recognizer.yml')

img = cv.imread('Faces/val/ben_afflek/5.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

face_react = haar_cascade.detectMultiScale(gray, 1.1, 4)
(x, y, w, h) = face_react[0]

face_roi = gray[y:y+h, x:x+h]

label, confidence = face_recognizer.predict(face_roi)

print(f'Predicted {people[label]} with a confidence of {confidence:.2f}%')

cv.putText(img, str(people[label]), (5, 30), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 1)
cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)

cv.imshow('Recognized face', img)


cv.waitKey(0)

