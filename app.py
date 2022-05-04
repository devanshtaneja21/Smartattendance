import cv2
import numpy as np
import face_recognition
import pickle
import datetime
from datetime import datetime
from Face_Rec_Class import Face_Rec
import time
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from tkinter import *
from PIL import Image,ImageTk
import threading

try:
    cred = credentials.Certificate("firebase_cred.json")
    firebase_admin.initialize_app(cred,{'databaseURL':"https://smartattendance-59a5b-default-rtdb.asia-southeast1.firebasedatabase.app/"})
except:
    print("Init Skipped")

def getDateTime():
    now = datetime.now()
    dtString = now.strftime('%Y_%m_%d_%H_%M_%S')
    return dtString


def markAttendance(name):
     with open('Attendance.csv', 'a+') as f:
                 now = datetime.now()
                 dtString = now.strftime('%d/%m/%Y %H:%M:%S')
                 f.writelines(f'\n{name},{dtString}')
def main():
    def on_close():
        running = False
        win.destroy()
        cap.release()
    def keyPress(event):
        if event.char.lower() == 'q':
            on_close()
    def sf():
        x = scale_factor_var.get().strip()
        if x.isnumeric:
            SCALE_FACTOR = float(x)
    def rf():
        x = recognize_frame_var.get().strip()
        if x.isnumeric:
            RECOGNIZE_FRAME_COUNT = int(x)
    def rt():
        x = restart_var.get().strip()
        if x.isnumeric:
            RESTART_TIME_SEC = int(x)
    def sngl():
          x = restart_var.get().strip().lower()
          if x=="true" or x=="True" or x=="TRUE":
              SINGLE_FACE = True
          if x=="false" or x=="False" or x=="FALSE":
              SINGLE_FACE = False
            
    win = Tk()
    win.geometry("840x480")
    win.protocol("WM_DELETE_WINDOW",on_close)
    win.bind('<KeyPress>',keyPress)
    win.configure(bg="black")
    f1=LabelFrame(win,bg="red")
    f1.place(x=0,y=0,width=640,height=480)
    L1=Label(f1,bg="red")
    L1.pack()
    scale_factor_var = StringVar()
    recognize_frame_var = StringVar()
    restart_var = StringVar()
    single_frame_var= StringVar()
    
    label1 = Label(win,text="Scale factor",font=('calibre',10,'bold'))
    label1_entry = Entry(win,textvariable=scale_factor_var,font=('calibre',10,'normal'))
    label1_entry.insert(0,"0.25")
    label1_button = Button(win,text="->",command=sf)
    label1.place(x=660,y=20)
    label1_entry.place(x=760,y=20,width=40)
    label1_button.place(x=800,y=20)
    
    label2 = Label(win,text="Frames",font=('calibre',10,'bold'))
    label2_entry = Entry(win,textvariable=recognize_frame_var,font=('calibre',10,'normal'))
    label2_entry.insert(0,"5")
    label2_button = Button(win,text="->",command=rf)
    label2.place(x=660,y=60)
    label2_entry.place(x=760,y=60,width=40)
    label2_button.place(x=800,y=60)
    
    label3 = Label(win,text="Restart Time",font=('calibre',10,'bold'))
    label3_entry = Entry(win,textvariable=restart_var,font=('calibre',10,'normal'))
    label3_entry.insert(0,"60")
    label3_button = Button(win,text="->",command=rt)
    label3.place(x=660,y=120)
    label3_entry.place(x=760,y=120,width=40)
    label3_button.place(x=800,y=120)
    
    label4 = Label(win,text="Single",font=('calibre',10,'bold'))
    label4_entry = Entry(win,textvariable=single_frame_var,font=('calibre',10,'normal'))
    label4_entry.insert(0,"true")
    label4_button = Button(win,text="->",command=sngl)
    label4.place(x=660,y=180)
    label4_entry.place(x=760,y=180,width=40)
    label4_button.place(x=800,y=180)
    
    
    
    SCALE_FACTOR = 0.25
    SINGLE_FACE = True
    RECOGNIZE_FRAME_COUNT = 5
    RESTART_TIME_SEC = 86400
    
    ref = db.reference("/Attendance")
    dt = getDateTime()
    mdt1=str(dt[5])
    mdt2=str(dt[6])
    mth=mdt1+mdt2
    print(mth)
    ref.child(dt).set(-1)
    ref = db.reference("/Attendance/"+dt)
    mref = db.reference("/Months/"+mth)
    START_TIME = time.time()
    cap = cv2.VideoCapture(0)
    face_recognizer = Face_Rec(TRAINING_FILE="training_encoding.pkl")
    FRAME_COUNT_DICT = {}
    ALREADY_UPDATED = {}
    ALREADY_UPDATED_LIST = []
    running =True
    while running:
        success, img = cap.read()
        if not success:
            print("Can't open Web camera. Exiting!")
            break
        
        _img = face_recognizer.resize(img,scale_factor=SCALE_FACTOR)
        _img = face_recognizer.cvt2rgb(_img)
    
        face_data = face_recognizer.find_faces(_img,single_face=SINGLE_FACE)
        if len(face_data) > 0:    
            for name,(x1,y1,x2,y2) in face_data:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                if name in ALREADY_UPDATED: continue
                if name in FRAME_COUNT_DICT: 
                    FRAME_COUNT_DICT[name] += 1
                else:
                    FRAME_COUNT_DICT[name] = 1
            
        for key in FRAME_COUNT_DICT:
            if FRAME_COUNT_DICT[key] >= RECOGNIZE_FRAME_COUNT and name not in ALREADY_UPDATED:
                markAttendance(name)
                ref.child(name).set(getDateTime())
                mref.child(name).push(getDateTime())
                
                ALREADY_UPDATED[name] = True
                ALREADY_UPDATED_LIST.append(name)
                print(f"Attendance {key}")
                
        for key in ALREADY_UPDATED_LIST:
            if key in FRAME_COUNT_DICT:
                del FRAME_COUNT_DICT[key]
                
        #cv2.imshow("WEBCAM",img)
        im=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))
        L1['image']=im
        win.update()
        try: 
            pass
        except KeyboardInterrupt:
            print("Int")
            break
        if abs(time.time() - START_TIME > RESTART_TIME_SEC):
            cap.release()
            win.destroy()
            main()
    
    cap.release()
    #cv2.destroyAllWindows()
main()