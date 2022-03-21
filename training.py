import cv2
import numpy as np
import face_recognition
import os
import pickle


path = 'Training_test'
encodeList = []
nameList = []
myList = os.listdir(path)
for cl in myList:
    img = cv2.imread(f'{path}/{cl}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encode = face_recognition.face_encodings(img)
    if len(encode) >= 1:
        encodeList.append(encode[0])      
        nameList.append(os.path.splitext(cl)[0])
    else:
        print(f'No encoding for {cl} found!')

#Saving the encodings and corresponding name list in a pkl file
with open("training_encoding.pkl","wb") as f:
    pickle.dump(encodeList,f)
    pickle.dump(nameList,f)

print("Training Done!")