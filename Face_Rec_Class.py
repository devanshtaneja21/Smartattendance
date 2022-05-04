import cv2
import numpy as np
import face_recognition
import os
import pickle


class Face_Rec:
    def __init__(self, TRAINING_FILE):
        with open(TRAINING_FILE,"rb") as f:
            self.encodeListKnown = pickle.load(f)
            self.nameListKnown = pickle.load(f)
            self.SCALE_FACTOR=1

    def resize(self,image,scale_factor=1):
        self.SCALE_FACTOR=scale_factor
        return cv2.resize(image, (0, 0), None, scale_factor, scale_factor)
    
    def cvt2rgb(self,image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def find_faces(self,img,single_face=True):
    
        data = [] #[(name,(x1,y1,x2,y2)) , ....]
        facesCurFrame = face_recognition.face_locations(img)
        encodesCurFrame = face_recognition.face_encodings(img, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = self.nameListKnown[matchIndex]
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = int(y1 // self.SCALE_FACTOR), int(x2 // self.SCALE_FACTOR), int(y2 // self.SCALE_FACTOR), int(x1 // self.SCALE_FACTOR)
                data.append((name,(x1,y1,x2,y2)))
                if single_face: break
        
        return data

    

