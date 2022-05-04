import cv2
import numpy as np
import face_recognition
import os
import pickle
import requests
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

try:
    cred = credentials.Certificate("firebase_cred.json")
    firebase_admin.initialize_app(cred,{'databaseURL':"https://smartattendance-59a5b-default-rtdb.asia-southeast1.firebasedatabase.app/",
                                        })
except:
    print("Init Skipped")

path = 'Training_folder'

existing_photos = os.listdir(path)

print("Fetching Image Please Wait...")

ref = db.reference("/Employees")
data = ref.get()
for key in data:
    name = data[key]['name'].strip()
    id = data[key]['id'].strip()
    url = data[key]['photo']
    image_name = name+"_"+id+".jpg"
    if image_name in existing_photos:
        continue
    img_data = requests.get(url).content
    with open(path+"/"+image_name,"wb") as f:
        f.write(img_data)

print("Images Fetched")
print("Training Started...")

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
