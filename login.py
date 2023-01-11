import tkinter
from tkinter import *
import tkinter as tk
from tkinter import Message ,Text
import cv2,os
import tkinter.ttk as ttk
import tkinter.font as font
import numpy as np
from PIL import Image, ImageTk
import csv
import json
import pandas as pd


#import shutil
#import pandas as pd
#import datetime
import time



window=tk.Tk()
window.title("Login Page")
window.geometry('1366x768')


   
####Background image
image1 = Image.open("face.png")
test = ImageTk.PhotoImage(image1)

label1 = tkinter.Label(window,image=test)
label1.image = test
label1.pack(fill=BOTH,expand=YES)


#####Loading Data from txt file
def loading_data():
    file=open('data.txt','r',encoding='utf-8')
    data=json.load(file)
    file.close()
    return data
##Saving Data to .txt file
def saving_data(data):
    file=open('data.txt','w',encoding='utf-8')
    json.dump(data,file,ensure_ascii=False)
    file.close()

##saving data in a variable
#data=dict()
data=loading_data()
print(data)


#HEADINGS
login_label= tk.Label(window, text="AUTHENTICATE YOURSELF!" ,bg="#000f1e"  ,fg="white"  ,width=25 ,height=1,font=('times', 30, 'bold')) 
login_label.place(x=340, y=10)



msg_label=tk.Label(window,text="AUTHENTICATION MESSAGE: ",bg='#001020' ,fg='white',width=26,height=1,font=('times',15,'bold'))
msg_label.place(x=150,y=500)
message = tk.Label(window,text="",fg="black", width=30, height=1, font=('times',15,'bold'))
message.place(x=480,y=500)



#Login FIEDLS
lbl = tk.Label(window, text="UNIQUE ID :",width=10,height=1  ,fg="#00192d" ,font=('times', 15, ' bold ') ) 
lbl.place(x=150, y=250)

txt = tk.Entry(window,width=25  ,bg="#003a62" ,show='*' ,fg="white",font=('times', 15, ' bold '))
txt.place(x=320, y=250)

lbl2= tk.Label(window, text="Enter Name :",width=11  ,height=1  ,fg="white"  ,bg="#001020" ,font=('times', 15, ' bold ') ) 
lbl2.place(x=150, y=300)

txt2 = tk.Entry(window,width=25  ,fg="#00192d",font=('times', 15, ' bold '))
txt2.place(x=320, y=300)


#Main

def login_clear():
    txt.delete(0,'end')
    txt2.delete(0,'end')
    #res = ""
    #txt.configure(text= res)
    #txt2.configure(text=res)

    #res = ""
    #txt3.configure(text= res)
    #txt4.configure(text=res)



################## Login_Functions   ######################

#When Submit button is clicked We have to Track the Person from our data through web cam
    
def TrackImages(UserId):
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);
    df=pd.read_csv("Details\Details.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX          
    run_count=0;run=True
    while run:
        
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
            print(Id, conf)
            if(conf < 50):
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                if (str(Id)==UserId):
                    message.configure(text="Face Recognised Successfully")
                    print("Hello "+aa)
                    message.configure(text="Hello "+aa)
                    run=False
            else:
                Id='Unknown'                
                tt=str(Id)            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        run_count += 1    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q') or run_count==150):
            message.configure(text="Unable to Recognise Face")
            break
    
    cam.release()
    cv2.destroyAllWindows()
    


def login_submit():
    a=txt.get()
    b=txt2.get()
    if (a in data):
        if(data[a] == b):
            TrackImages(a)
        else:
            message.configure(text="Id and Password does not Match")
    else:
        message.configure(text="Entered Id does not Exists")

    login_clear()




#Login Actions
submit = tk.Button(window, text="Submit",fg="white",command= login_submit, bg="#001020"  ,width=25  ,height=1 ,relief="groove" ,activebackground = "white" ,font=('times', 10, ' bold '))
submit.place(x=250, y=350)


clearButton = tk.Button(window, text="Clear",fg="#001020", command=login_clear,bg="white"  ,width=25  ,height=1, relief='groove', activebackground = "blue" ,activeforeground='white',font=('times', 10, ' bold '))
clearButton.place(x=450, y=350)




#final Actions
quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg="white"  ,bg="red"  ,width=10  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=1050, y=550)
window.mainloop()
