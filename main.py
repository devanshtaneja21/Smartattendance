import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pickle

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')

#Scale factor for frame
SCALE_FACTOR = 0.25

#Loading training encodings from pickle file
with open("training_encoding.pkl","rb") as f:
    encodeListKnown = pickle.load(f)
    nameListKnown = pickle.load(f)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Can't open Cam. Exiting!")
        break
    
    #Resizing image and converting to RGB ColorSpace
    resized_img = cv2.resize(img, (0, 0), None, SCALE_FACTOR, SCALE_FACTOR)
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    #Finding location of face and the corresponding face_encoding on webcam
    facesCurFrame = face_recognition.face_locations(resized_img)
    encodesCurFrame = face_recognition.face_encodings(resized_img, facesCurFrame)

    #For every encoding found in webcam, compare with known encodings to match faces
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = nameListKnown[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 // SCALE_FACTOR, x2 // SCALE_FACTOR, y2 // SCALE_FACTOR, x1 // SCALE_FACTOR
            y1, x2, y2, x1 = int(y1), int(x2), int(y2), int(x1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1)==ord("q"):
        break 
cap.release()    
cv2.destroyAllWindows()